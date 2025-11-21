from typing import cast, Callable
from collections import defaultdict
from tinygrad.helpers import strip_parens
from tinygrad.device import Compiled, LRUAllocator, BufferSpec, Compiler
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import UOp, Ops, GroupOp, PatternMatcher, UPat, print_uops, graph_rewrite
from tinygrad.dtype import AddrSpace, dtypes, DType, PtrDType
from tinygrad.runtime.ops_python import PythonAllocator

from bgpu_assembler import BGPUAssembler
from bgpu_driver import BGPUDriver

import struct
import math
import functools

bgpu_widest_type = dtypes.int
bgpu_addr_type = dtypes.int

bgpu_global_size = 1 << 24
bgpu_local_size = 4
bgpu_max_registers = 256

emulate = False

class BGPUProgram:
  def __init__(self, device, name:str, lib:bytes): self.device, self.function_name, self.lib = device, name, lib

  def kernel(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), **kwargs): 
    print(f"{self.function_name}")
    print(f"{self.lib.decode("utf-8")}")
    print(f"running kernel with global_size={global_size}, local_size={local_size}")
    print(f"args={args}, kwargs={kwargs}")
    if local_size > (bgpu_local_size, 1, 1) or global_size > (bgpu_global_size, 1, 1):
      raise ValueError(f"global_size {global_size} or local_size {local_size} exceeds maximums ({bgpu_global_size}, 1, 1) and ({bgpu_local_size}, 1, 1)")

    # Compile the kernel
    program = BGPUAssembler().assemble_lines(self.lib.decode("utf-8").splitlines())

    # Run the kernel
    self.device.driver.run_kernel(args, global_size=global_size, local_size=local_size, program=program, function_name=self.function_name)

    return

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    return self.kernel(*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait)

def render_val(x, dtype):
  if dtypes.is_float(dtype):
    if dtype == dtypes.double: return "0d%02X%02X%02X%02X%02X%02X%02X%02X" % tuple(struct.pack("d",x)[::-1])
    if dtype == dtypes.half: return "0x%02X%02X" % tuple(struct.pack("e",x)[::-1])
    print(dtype)
    print(x)
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
  Ops.EXP2: lambda a: "",
  Ops.LOG2: lambda a: "",
  Ops.MAX: lambda a: "",
}

def mem_type(x: UOp): return 'global'

asm_rewrite = PatternMatcher([
  # Range with constant bound
  (UPat(Ops.RANGE, name="x", allow_any_len=True), lambda ctx,x: f"mov.ri.{ctx.types[x.dtype]}\t\t{ctx.r[x].rjust(4)}, {"0".rjust(4)} # init range\nloop_{ctx.r[x]}:"),

  # End Range
  (UPat(Ops.END, name="x", src=(UPat.var('range'), UPat.var('last_op'))), lambda ctx,x,range,last_op:
    [f"checkloop_{ctx.r[range]}:",
    f"\tadd.ri.{ctx.types[range.dtype]}\t\t{ctx.r[range].rjust(4)}, {ctx.r[range].rjust(4)}, {"1".rjust(4)} # increment",
    f"\tsub.rr.{ctx.types[range.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[range].rjust(4)}, {ctx.r[range.src[0]].rjust(4)} # compare bound",
    f"\tbr.nz.loop_{ctx.r[range]}\t\t{ctx.r[x].rjust(4)} # loop back if not done",
    f"endloop_{ctx.r[range]}:"]
  ),

  # Constants -> mov.ri
  (UPat.cvar("x"), lambda ctx, x: f"mov.ri.{ctx.types[x.dtype][0:]}\t{ctx.r[x].rjust(4)}, {render_val(x.arg, x.dtype)} # constant"),

  # Load with just a base address-> ld
  (UPat(Ops.LOAD, name="x", src=(UPat.var('base'))),
   lambda ctx, x, base: None \
     if x.dtype.count > 1 else f"ld.{ctx.types[x.dtype]}.{mem_type(x)}\t\t{ctx.r[x].rjust(4)}, {ctx.r[base].rjust(4)}"),

  # Gated index -> no-op as it is handled in a gated load
  (UPat(Ops.INDEX, name="x", src=(UPat.var("buf"), UPat.var("loc"), UPat.var("gate"))),
    lambda ctx, x, loc, gate, buf: f"# Gated index {ctx.r[x]}"),

  # Gated Load
  (UPat(Ops.LOAD, name="x", src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("loc"), UPat.var("gate"))), UPat.var("alt"))),
    lambda ctx, x, loc, alt, gate, buf: 
    None if x.dtype.count > 1 else [f"\tmov.rr.{ctx.types[x.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[alt].rjust(4)} # load alternative",
    f"\tbr.ez.load_{ctx.r[x]} {ctx.r[gate].rjust(4)} # if gate is zero, skip load",
    f"\tshl.ri.{ctx.types[bgpu_addr_type]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[loc].rjust(4)}, {render_val(math.log2(buf.dtype.base.scalar().itemsize), bgpu_addr_type)} # index shift",
    f"\tadd.rr.{ctx.types[bgpu_addr_type]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[buf].rjust(4)}, {ctx.r[x].rjust(4)} # index into buffer",
    f"\tld.{ctx.types[x.dtype]}.{mem_type(x)}\t\t{ctx.r[x].rjust(4)}, {ctx.r[x].rjust(4)} # load",
    f"load_{ctx.r[x]}: # skip label for load",
    "\tsync.threads"
    ]),

  # Where
  (UPat(Ops.WHERE, name="x", src=(UPat.var('cond'), UPat.var('a'), UPat.var('b'))),
    lambda ctx, x, cond, a, b:
    None if x.dtype.count > 1 else [
      f"\tmov.rr.{ctx.types[x.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[b].rjust(4)} # where false case",
      f"\tbr.ez.where_{ctx.r[x]} {ctx.r[cond].rjust(4)} # if cond is zero, skip true case",
      f"\tmov.rr.{ctx.types[x.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[a].rjust(4)} # where true case",
      f"where_{ctx.r[x]}: # skip label for where",
      "\tsync.threads"
      ]),

  # Store with a just a base address
  (UPat(Ops.STORE, name="x", src=(UPat.var('base'), UPat.var("var"))), lambda ctx, x, base, var:
    None if var.dtype.count > 1 or x.arg is not None else
    f"st.{ctx.types[var.dtype.scalar()]}.{mem_type(base)}\t\t" + \
    f"{ctx.r[base].rjust(4)}, {('{' + ', '.join(ctx.r[var]) + '}') if var.dtype.count > 1 else ctx.r[var].rjust(4)}"),

  # Store register into a register -> mov.rr
  (UPat(Ops.STORE, name="x", src=(UPat.var('base'), UPat.var("var"))), lambda ctx, x, base, var:
    None if var.dtype.count > 1 or x.arg is None or x.arg.op != Ops.DEFINE_REG else
    f"mov.rr.{ctx.types[var.dtype]}\t\t\t" + \
    f"{ctx.r[x.arg].rjust(4)}, {ctx.r[var].rjust(4)} # store register into register"),

  # ALU register register
  (UPat({Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE}, name="x", src=(UPat.var('a'), UPat.var('b'))),
   lambda ctx, x, a, b: f"{x.op.name.lower()}.rr.{ctx.types[a.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[a].rjust(4)}, {ctx.r[b].rjust(4)}"),

  # ALU register register
  (UPat(GroupOp.ALU, name="x", src=(UPat.var('a'), UPat.var('b'))),
   lambda ctx, x, a, b: f"{x.op.name.lower()}.rr.{ctx.types[x.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[a].rjust(4)}, {ctx.r[b].rjust(4)}"),

  # ALU register constant
  (UPat(GroupOp.ALU, name="x", src=(UPat.var('a'))),
   lambda ctx, x, a: None if x.arg == None else f"{x.op.name.lower()}.ri.{ctx.types[x.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[a].rjust(4)}, {render_val(x.arg, x.dtype).rjust(4)}"),

  # ALU register
  (UPat(GroupOp.ALU, name="x", src=(UPat.var('a'))),
   lambda ctx, x, a: f"{x.op.name.lower()}.rr.{ctx.types[x.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[a].rjust(4)}"),

  # Parameter
  (UPat(Ops.DEFINE_GLOBAL, name="x"), lambda ctx,x:
    f"ldparam.{ctx.types[bgpu_addr_type]} {ctx.r[x].rjust(4)}, {x.arg} # define global"),

  # Index
  (UPat(Ops.INDEX, name="x", src=(UPat.var('a'), UPat.var('b'))),
   lambda ctx, x, a, b: [
    f"\tshl.ri.{ctx.types[bgpu_addr_type]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[b].rjust(4)}, {render_val(math.log2(a.dtype.base.scalar().itemsize), bgpu_addr_type)} # index shift",
    f"\tadd.rr.{ctx.types[bgpu_addr_type]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[a].rjust(4)}, {ctx.r[x].rjust(4)} # index"
   ]),

  # Special
  (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: 
   f"special\t\t\t{ctx.r[x].rjust(4)}, %{x.arg[0]}"
  ),

  # Cast
  (UPat({Ops.CAST, Ops.BITCAST}, name="x", src=(UPat.var('a'))), lambda ctx,x,a:
    f"cast.{ctx.types[x.dtype]}.{ctx.types[a.dtype]}\t{ctx.r[x]}, {ctx.r[a]}"
  ),

  # Define Register
  (UPat(Ops.DEFINE_REG, name="x"), lambda ctx, x: f"mov.ri.{ctx.types[x.dtype.base.scalar()][0:]}\t{ctx.r[x]}, {x.arg[0]} # define register"),

  # Sink
  (UPat(Ops.SINK), lambda: "stop"),
]
)

class BGPURenderer(Renderer):
  device = "BGPU"
  suffix = ".bgpu"
  supports_float4 = False
  has_local = True
  has_threads = True
  has_shared = False
  global_max = (bgpu_global_size, 1, 1)
  local_max = (bgpu_local_size, 1, 1)
  shared_max = 0
  tensor_coes = []
  pre_matcher = None
  extra_matcher = None
  code_for_op = asm_for_op

  types: dict[DType, str] = {
    dtypes.void: "void",
    dtypes.char: "int8",
    dtypes.uchar: "uint8",
    dtypes.short: "int16",
    dtypes.int: "int32",
    dtypes.uint: "uint32",
    dtypes.bool: "bool",
    dtypes.float: "float32",
    dtypes.long: "long"
  }

  def render_kernel(self, function_name):
    kernel:list[str] = []

    for u in self.uops:
      # Render instructions as assembly
      if (l:=cast(str|list[str], asm_rewrite.rewrite(u, ctx=self))) is None:
        raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}\nKernel: {'\n'.join(kernel)}")

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
        # else:
        #   self.r_lifetime[self.r[u]] = (self.r_lifetime[self.r[u]][0], uop_idx)

      for src in u.src:
        if src not in self.r:
          print(f"Warning: source {src} not in registers")
          continue
        if self.r[src] not in self.r_lifetime:
          print(f"Warning: source {src} has no lifetime when used in uop {u}")
          continue
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

    # print("Legalizing uops...")
    # actual_uops = []
    # for u in uops:
    #   l = bgpu_legalizer.rewrite(u, ctx=self)
    #   if l is not None:
    #     actual_uops.append(l)
    #   else:
    #     actual_uops.append(u)
    # uops = actual_uops

    # print("Legalized uops:")
    print_uops(uops)

    print("Extracting register information...")
    actual_uops = []
    has_range = False
    for uop_idx, u in enumerate(uops):
      # Noop
      if u.op is Ops.NOOP:
        continue

      # Range
      if u.op is Ops.RANGE:
        has_range = True

      # Sink Op
      if u.op is Ops.SINK:
        if u.arg is not None:
          function_name = u.arg.function_name
        actual_uops.append(UOp(Ops.SINK))
        continue

      # After
      if u.op is Ops.AFTER:
        r[u] = r[u.src[0]]
        continue

      # Casts into same type or pointer casts are no-ops
      if u.op in {Ops.CAST, Ops.BITCAST} and (u.src[0].dtype == u.dtype or isinstance(u.src[0].dtype, PtrDType) or u.src[0].op is Ops.SPECIAL):
        r[u] = r[u.src[0]]
        continue

      # Cast from unsigned to signed of same size is no-op
      if u.op is Ops.CAST and dtypes.is_unsigned(u.src[0].dtype) != dtypes.is_unsigned(u.dtype) and u.src[0].dtype.itemsize == u.dtype.itemsize:
        r[u] = r[u.src[0]]
        continue

      # Cast to/from long is no-op
      if u.op is Ops.CAST and (u.src[0].dtype == dtypes.long or u.dtype == dtypes.long):
        r[u] = r[u.src[0]]
        continue

      # Back-to-back casts of the same type: intX -> intY -> intX
      if u.op is Ops.CAST and u.src[0].op is Ops.CAST and u.dtype == u.src[0].src[0].dtype:
        r[u] = r[u.src[0].src[0]]
        continue

      # Cast from bool to int is a no-op
      if u.op is Ops.CAST and u.src[0].dtype == dtypes.bool and dtypes.is_integer(u.dtype):
        r[u] = r[u.src[0]]
        continue

      # Index with an after pointing to a register
      if u.op is Ops.INDEX and u.src[0].op is Ops.AFTER and u.src[0].src[0].op is Ops.DEFINE_REG:
        u.src = [u.src[0].src[0], u.src[1]] # Replace AFTER with DEFINE_REG
        print(f"index after into register {u.src[0]}")

      # Index 0 into register is just the register itself
      if u.op is Ops.INDEX and u.arg == None and u.src[0].op is Ops.DEFINE_REG and u.src[1].op is Ops.CONST and u.src[1].arg == 0:
        # Use the register directly
        print(f"index 0 into register {u.src[0]} -> using register directly")
        r[u] = r[u.src[0]]
        continue

      # Load into index of register is just a move -> use the same register
      if u.op is Ops.LOAD and u.src[0].op is Ops.INDEX and u.src[0].src[0].op is Ops.DEFINE_REG:
        r[u] = r[u.src[0].src[0]]
        print(f"load from index into register {u.src[0].src[0]}: {r[u]}")
        continue

      # # Store into register
      if u.op is Ops.STORE and u.src[0].op is Ops.INDEX and u.src[0].src[0].op is Ops.DEFINE_REG:
        # u.src = [u.src[0].src[0], u.src[1]]  # Replace index with the DEFINE_REG
        u.arg = u.src[0].src[0]  # Destination register
        r[u] = r[u.src[0]] # Value is the value to store

      # if u.op in {Ops.ADD, Ops.SUB, Ops.MUL, Ops.SHL, Ops.SHR, Ops.IDIV}:
      #   # It is an ALU operation
      #   if len(u.src) > 1 and u.src[1].op == Ops.CONST:
      #     # Remove the constant from src[1] and put it into the arguments
      #     u.arg = u.src[1].arg
      #     u.src = (u.src[0],)  # Remove the constant from src

      # Load parameter -> need to load the parameter base address first
      # if u.op is Ops.DEFINE_GLOBAL:
      #   param_base_address_register = "param_base_address"
      #   if not param_address_loaded:
      #     load_base_address = UOp(Ops.SPECIAL, arg=('param',), dtype=bgpu_addr_type)
      #     r[load_base_address] = param_base_address_register
      #     actual_uops.append(load_base_address)
      #     param_address_loaded = True

      #   u.src = (load_base_address, )  # Add the base address as operand

      actual_uops.append(u)

      if u.op is Ops.SPECIAL:
        print(f"special register {u.arg[0]}")
        r[u] = "%" + u.arg[0]
        continue
      elif u.op is Ops.LOAD:
        # assert u.src[0].dtype == bgpu_addr_type, f"address of load isn't {bgpu_addr_type} but {u.src[0].dtype}"
        r[u] = [ssa('val', dtype=self.types[u.dtype.scalar()]) for _ in range(u.dtype.count)] if u.dtype.count > 1 else ssa('val', u)
        continue
      elif u.op is Ops.DEFINE_GLOBAL: 
        print(f"global {u.arg} of type {u.dtype}")
        bufs.append((f"data{u.arg}", u.dtype))
      
      prefix, dtype = {
          Ops.END: ("range_cond", self.types[dtypes.int]),
          Ops.RANGE: ("range", self.types[u.dtype.base.scalar()]),
          Ops.INDEX: ("idx", self.types[u.dtype.base.scalar()]),
          Ops.CAST: ("cast", None),
          Ops.BITCAST: ("cast", None),
          Ops.CONST: ("const", None),
          Ops.DEFINE_GLOBAL: ("param", self.types[bgpu_widest_type]),
          Ops.DEFINE_REG: ("reg", self.types[u.dtype.base.scalar()]),
          **{op: ("alu", None) for op in GroupOp.ALU}
        }.get(u.op, (None, None))
      if prefix:
        r[u] = ssa(prefix, u, dtype)
        print(f"assigned register {r[u]} for uop {u.op}")
      else:
        print(f"Warning: no register assigned for uop {u.op}")

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
        if self.r_lifetime[r[u]][0] == -1:
          raise RuntimeError(f"register {r[u]} has no start lifetime for uop {u}")

        if self.r_lifetime[r[u]][1] == -1:
          if "range_cond_" not in r[u]:
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

      if not has_range:
      # TODO: This does not work if there are loops -> we just never free registers
      # Check if we can free a register
      # If the register is used last in current uop, then we can use it as destination for this uop
        for free_reg in ar_lifetime:
          if ar_lifetime[free_reg] == uop_idx:
            print(f"freeing {free_reg}")
            ar_lifetime[free_reg] = -1

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

  def __getitem__(self, key): return "", ""

class BGPUAllocator(LRUAllocator['BGPUDevice']):
  def _alloc(self, size, options:BufferSpec):
    print(f"allocating {size} bytes")
    print(f"options: {options}")
    assert not options.uncached, "BGPU does not support uncached allocations"
    assert not options.cpu_access, "BGPU does not support CPU access allocations"
    assert not options.host, "BGPU does not support host allocations"
    assert not options.nolru, "BGPU does not support nolru allocations"
    return self.dev.driver.alloc(size)

  def _free(self, opaque, options:BufferSpec):
    print(f"freeing {opaque}")
    print(f"options: {options}")
    assert False, "BGPU free not implemented"

  def _copyin(self, dest, src:memoryview):
    print(f"copying in {len(src)} bytes")
    print(f"to {dest}")
    self.dev.driver.copy_h2d(dest, src)

  def _copyout(self, dest:memoryview, src):
    print(f"copying out {len(dest)} bytes")
    print(f"from {src}")
    self.dev.driver.copy_d2h(dest, src)

  def _transfer(self, dest, src, sz:int, src_dev, dst_dev):
    print(f"transferring {sz} bytes on device from {src_dev} to {dst_dev}")
    assert False, "BGPU transfer not implemented"

class BGPUDevice(Compiled):
  def __init__(self, device:str):
    self.driver = BGPUDriver()
    super().__init__(device, PythonAllocator(self) if emulate else BGPUAllocator(self), [(BGPURenderer, Compiler)], functools.partial(BGPUProgram, self), None)

