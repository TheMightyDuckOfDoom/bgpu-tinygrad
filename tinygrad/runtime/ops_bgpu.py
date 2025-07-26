from typing import cast, Callable
from collections import defaultdict
from tinygrad.helpers import strip_parens
from tinygrad.device import Compiled, LRUAllocator, BufferSpec
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import UOp, Ops, GroupOp, PatternMatcher, UPat, print_uops, graph_rewrite
from tinygrad.dtype import dtypes, DType, PtrDType
from tinygrad.runtime.ops_python import PythonAllocator
import struct
import math

bgpu_widest_type = dtypes.int
bgpu_addr_type = dtypes.int

bgpu_global_size = 256
bgpu_local_size = 4
bgpu_max_registers = 256

emulate = False

class BGPUProgram:
  def __init__(self): pass

  def kernel(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), **kwargs): 
    print(f"{self.function_name}")
    print(f"{self.lib.decode("utf-8")}")
    print(f"running kernel with global_size={global_size}, local_size={local_size}")
    print(f"args={args}, kwargs={kwargs}")
    if local_size > (bgpu_local_size, 1, 1) or global_size > (bgpu_global_size, 1, 1):
      raise ValueError(f"global_size {global_size} or local_size {local_size} exceeds maximums ({bgpu_global_size}, 1, 1) and ({bgpu_local_size}, 1, 1)")
    return

  def __call__(self, function_name:str, lib:bytes): 
    self.function_name = function_name
    self.lib = lib
    return self.kernel

def render_val(x, dtype):
  if dtypes.is_float(dtype):
    if dtype == dtypes.double: return "0d%02X%02X%02X%02X%02X%02X%02X%02X" % tuple(struct.pack("d",x)[::-1])
    if dtype == dtypes.half: return "0x%02X%02X" % tuple(struct.pack("e",x)[::-1])
    return "0f%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])
  return str(int(x)) + ("U" if dtypes.is_unsigned(dtype) else "")

asm_for_op: dict[Ops, Callable] = {
  Ops.AND: lambda d,a,b,dt, name: f"and.pred {d}, {a}, {b};" if dt == dtypes.bool else f"and.b{name[1:]} {d}, {a}, {b}",
  Ops.OR: lambda d,a,b,dt, name: f"or.pred {d}, {a}, {b};" if dt == dtypes.bool else f"or.b{name[1:]} {d}, {a}, {b}",
  Ops.ADD: lambda d,a,b,dt,name: f"{'or' if dt == dtypes.bool else 'add'}.{name} {d}, {a}, {b}",
  Ops.SUB: lambda d,a,b,name: f"sub.{name} {d}, {a}, {b}",
  Ops.MUL: lambda d,a,b,dt,name: f"{'and' if dt == dtypes.bool else 'mul'}.{name} {d}, {a}, {b}",
  Ops.SHL: lambda d,a,b,name: f"shl.{name[1:]} {d}, {a}, {b}",
  Ops.SHR: lambda d,a,b,name: f"shr.{name[1:]} {d}, {a}, {b}",
}

def mem_type(x: UOp): return 'global'

asm_rewrite = PatternMatcher([
  # Constants -> mov.imm
  (UPat.cvar("x"), lambda ctx, x: f"mov.imm.{ctx.types[x.dtype][0:]}\t{ctx.r[x]}, {render_val(x.arg, x.dtype)}"),

  # Load with just a base address-> ld
  (UPat(Ops.LOAD, name="x", src=(UPat.var('base')), allow_any_len=True),
   lambda ctx, x, base: None \
     if x.dtype.count > 1 else f"ld.{ctx.types[x.dtype]}.{mem_type(x)}\t\t{ctx.r[x].rjust(4)}, {ctx.r[base].rjust(4)}"),

  # Store with a just a base address
  (UPat(Ops.STORE, src=(UPat.var('base'), UPat.var("var")), allow_any_len=True), lambda ctx, base, var:
    None if var.dtype.count > 1 else
    f"st.{ctx.types[var.dtype.scalar()]}.{mem_type(base)}\t\t" + \
    f"{ctx.r[base].rjust(4)}, {('{' + ', '.join(ctx.r[var]) + '}') if var.dtype.count > 1 else ctx.r[var].rjust(4)}"),

  # MUL register constant
  (UPat(Ops.MUL, name="x", src=(UPat.var('a'))),
    lambda ctx, x, a: f"shl.ri.{ctx.types[x.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[a].rjust(4)}, {render_val(math.log2(x.arg), x.dtype).rjust(4)}" if x.arg.bit_count() == 1 else None),

  # ALU register register
  (UPat(GroupOp.ALU, name="x", src=(UPat.var('a'), UPat.var('b')), allow_any_len=True),
   lambda ctx, x, a, b: f"{x.op.name.lower()}.rr.{ctx.types[x.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[a].rjust(4)}, {ctx.r[b].rjust(4)}"),

  # ALU register constant
  (UPat(GroupOp.ALU, name="x", src=(UPat.var('a')), allow_any_len=True),
   lambda ctx, x, a: f"{x.op.name.lower()}.ri.{ctx.types[x.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[a].rjust(4)}, {render_val(x.arg, x.dtype).rjust(4)}"),

  # Parameter
  (UPat(Ops.DEFINE_GLOBAL, name="x", src=(UPat.var('param'))), lambda ctx,x, param:
    [f"\tadd.ri.{ctx.types[bgpu_addr_type]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[param].rjust(4)}, {x.arg * 4}",
      f"\tld.{ctx.types[param.dtype]}.global\t\t{ctx.r[x].rjust(4)}, {ctx.r[x].rjust(4)}"]
    if x.arg != 0 else
    f"ld.{ctx.types[param.dtype]}.global\t\t{ctx.r[x].rjust(4)}, {ctx.r[param].rjust(4)}"),

  # Index
  (UPat(Ops.INDEX, name="x", src=(UPat.var('a'), UPat.var('b'))),
   lambda ctx, x, a, b: f"add.rr.{ctx.types[bgpu_addr_type]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[a].rjust(4)}, {ctx.r[b].rjust(4)}"),

  # Special
  (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: 
   f"special\t\t\t{ctx.r[x].rjust(4)}, %{x.arg[0]}"
  ),

  # Sink
  (UPat(Ops.SINK), lambda: "stop"),
]
)

class BGPURenderer(Renderer):
  device = "BGPU"
  suffix = ".bgpu"
  has_shared = True
  has_local = True
  shared_max = 0
  supports_float4 = False
  special_loopidx_dtype = bgpu_widest_type
  global_max = (bgpu_global_size, 1, 1)
  local_max = (bgpu_local_size, 1, 1)

  types: dict[DType, str] = { dtypes.char: "int8", dtypes.uchar: "uint8", dtypes.short: "int16", dtypes.int: "int32", dtypes.uint: "uint32" }

  def render_kernel(self, function_name):
    kernel:list[str] = []

    for u in self.uops:
      # Render instructions as assembly
      if (l:=cast(str|list[str], asm_rewrite.rewrite(u, ctx=self))) is None:
        raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")

      if l is not None and not (isinstance(l, str) and l == ""):
        kernel.extend(['\t' + l] if isinstance(l, str) else l)

    kernel = '\n'.join(kernel)
    kernel = function_name + ":\n" + kernel
    return kernel

  def calculate_lifetimes(self):
    self.r_lifetime = {}
    for uop_idx, u in enumerate(self.uops):
      if u in self.r:
        if self.r[u] not in self.r_lifetime:
          self.r_lifetime[self.r[u]] = (uop_idx, -1)
        else:
          self.r_lifetime[self.r[u]] = (self.r_lifetime[self.r[u]][0], uop_idx)

      for src in u.src:
        if self.r[src] not in self.r_lifetime:
          self.r_lifetime[self.r[src]] = (-1, uop_idx)
        else:
          self.r_lifetime[self.r[src]] = (self.r_lifetime[self.r[src]][0], uop_idx)

  def render(self, uops:list[UOp]) -> str:
    print("Rendering BGPU code")

    function_name = "BGPU_KERNEL"

    bufs = []

    # Holds how many of which register we have
    c: defaultdict[str, int] = defaultdict(int)
    
    # Maps UOp to register name
    r: dict[UOp, str] = {}

    self.r = r
    self.c = c
    self.uops = uops
    self.r_lifetime: dict[UOp, tuple(int, int)|None] = {}

    def ssa(prefix:str, u:UOp|None=None, dtype:str|None=None) -> str:
      nonlocal c, r
      # print(f"ssa({prefix}, {u}, {dtype})")
      prefix += f"_{dtype if dtype is not None else self.types[cast(UOp, u).dtype]}_"
      c[prefix] += 1
      return f"%{prefix}{c[prefix]-1}"

    print("\nStarting to render")

    print("Legalizing uops...")
    # actual_uops = []
    # for u in uops:
    #   l = bgpu_legalizer.rewrite(u, ctx=self)
    #   if l is not None:
    #     actual_uops.append(l)
    #   else:
    #     actual_uops.append(u)
    # uops = actual_uops

    print("Legalized uops:")
    print_uops(uops)

    print("Extracting register information...")
    actual_uops = []
    param_address_loaded = False
    load_base_address = None
    for uop_idx, u in enumerate(uops):
      if u.op is Ops.SINK:
        if u.arg is not None:
          function_name = u.arg.function_name
        actual_uops.append(UOp(Ops.SINK))
        continue

      if u.op in {Ops.CAST, Ops.BITCAST} and (u.src[0].dtype == u.dtype or isinstance(u.src[0].dtype, PtrDType) or u.src[0].op is Ops.SPECIAL):
        # Casts into same type or pointer casts are no-ops
        r[u] = r[u.src[0]]
        continue

      if u.op is Ops.CAST and u.src[0].op is Ops.CAST and u.dtype == u.src[0].src[0].dtype:
        # Back-to-back casts of the same type: intX -> intY -> intX
        r[u] = r[u.src[0].src[0]]
        continue

      if u.op in asm_for_op:
        # It is an ALU operation
        if u.src[1].op == Ops.CONST:
          # Remove the constant from src[1] and put it into the arguments
          u.arg = u.src[1].arg
          u.src = (u.src[0],)  # Remove the constant from src

      # Load parameter -> need to load the parameter base address first
      if u.op is Ops.DEFINE_GLOBAL:
        param_base_address_register = "param_base_address"
        if not param_address_loaded:
          load_base_address = UOp(Ops.SPECIAL, arg=('param', ), dtype=bgpu_addr_type)
          r[load_base_address] = param_base_address_register
          actual_uops.append(load_base_address)
          param_address_loaded = True

        u.src = (load_base_address, )  # Add the base address as operand

      actual_uops.append(u)

      if u.op is Ops.SPECIAL:
        print(f"special register {u.arg[0]}")
        r[u] = "%" + u.arg[0]
      elif u.op is Ops.LOAD:
        # assert u.src[0].dtype == bgpu_addr_type, f"address of load isn't {bgpu_addr_type} but {u.src[0].dtype}"
        r[u] = [ssa('val', dtype=self.types[u.dtype.scalar()]) for _ in range(u.dtype.count)] if u.dtype.count > 1 else ssa('val', u)
      elif u.op is Ops.DEFINE_GLOBAL: 
        print(f"global {u.arg} of type {u.dtype}")
        bufs.append((f"data{u.arg}", u.dtype))

      prefix, dtype = {
          Ops.INDEX: ("idx", u.dtype),
          Ops.CAST: ("cast", None),
          Ops.BITCAST: ("cast", None),
          Ops.CONST: ("const", None),
          Ops.DEFINE_GLOBAL: ("param", self.types[bgpu_widest_type]),
          **{op: ("alu", None) for op in GroupOp.ALU}
        }.get(u.op, (None, None))
      if prefix:
        r[u] = ssa(prefix, u, dtype)

      print_uops([u])
    self.uops = actual_uops

    print("Uops after extracting register information:")
    print_uops(self.uops)

    print("Initial lifetimes:")
    self.calculate_lifetimes()
    for src in self.r_lifetime:
      print(f"{src}: {self.r_lifetime[src]}")

    # Remove Ops with unused results
    print("Removing unused results...")
    actual_uops = []
    for u in self.uops:
      if u in r:
        assert(r[u] in self.r_lifetime) # Has to have a lifetime
        assert(self.r_lifetime[r[u]][0] != -1) # Has to have a start

        if self.r_lifetime[r[u]][1] == -1:
          # Never used
          print(f"never used {r[u]}")
          continue
      actual_uops.append(u)
    self.uops = actual_uops

    print(self.render_kernel(function_name))

    print("Lifetimes after removing unused results:")
    self.calculate_lifetimes()
    for src in self.r_lifetime:
      print(f"{src}: {self.r_lifetime[src]}")

    # Allocate registers
    print("Allocating registers...")
    # Maps register name to architectural register
    ar_lifetime: dict[int, int] = {}
    r_to_ar: dict[str, int] = {}

    # Regs in lifetime are already sorted by start
    uop_idx = 0
    reg_idx = 0
    reg_to_schedule = list(self.r_lifetime)[reg_idx]
    assert(self.r_lifetime[reg_to_schedule][0] == 0) # First register has to start at 0
    while(len(r_to_ar) < len(self.r_lifetime)):
      # Check if we can free a register
      # If the register is used last in current uop, then we can use it as destination for this uop
      for free_reg in ar_lifetime:
        if ar_lifetime[free_reg] == uop_idx:
          print(f"freeing {free_reg}")
          ar_lifetime[free_reg] = -1

      # Schedule the register
      if reg_idx >= len(self.r_lifetime):
        raise RuntimeError(f"index {reg_idx} out of bounds {len(self.r_lifetime)}")
      reg_to_schedule = list(self.r_lifetime)[reg_idx]
      if uop_idx == self.r_lifetime[reg_to_schedule][0]:
        # Find first free register
        for i in range(bgpu_max_registers + 1):
          if i not in ar_lifetime or ar_lifetime[i] == -1:
            print(f"scheduling {reg_to_schedule} to r{i}")
            ar_lifetime[i] = self.r_lifetime[reg_to_schedule][1] # set end
            r_to_ar[reg_to_schedule] = i # map register to architectural register
            break
          assert(i != bgpu_max_registers) # No free registers
        # Goto next register
        reg_idx += 1

      uop_idx += 1

    # Print register allocation
    for reg in r_to_ar:
      print(f"{reg} -> r{r_to_ar[reg]}")

    # Rewrite the registers to use architectural registers
    new_r: dict[UOp, str] = {}
    for u in self.r:
      if self.r[u] in r_to_ar:
        new_r[u] = f"r{r_to_ar[self.r[u]]}"
    self.r = new_r

    # Render the uops
    return self.render_kernel(function_name)

class BGPUAllocator(LRUAllocator['BGPUDevice']):
  def _alloc(self, size, options:BufferSpec): print(f"allocating {size} bytes")
  def _free(self, opaque, options:BufferSpec): print(f"freeing {opaque}")
  def _copyin(self, dest, src:memoryview): print(f"copying in {len(src)} bytes")
  def _copyout(self, dest:memoryview, src): print(f"copying out {len(dest)} bytes")
  def _transfer(self, dest, src, sz:int, src_dev, dst_dev): print(f"transferring {sz} bytes from {src_dev} to {dst_dev}")

class BGPUDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, PythonAllocator() if emulate else BGPUAllocator(self), BGPURenderer(), None, BGPUProgram(), None)
