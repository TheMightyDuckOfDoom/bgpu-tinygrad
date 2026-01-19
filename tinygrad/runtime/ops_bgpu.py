from typing import cast, Callable
from collections import defaultdict
from tinygrad.device import Compiled, LRUAllocator, BufferSpec, Compiler, CompilerSet, CompilerPair
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import UOp, Ops, GroupOp, PatternMatcher, UPat, print_uops
from tinygrad.dtype import dtypes, DType, PtrDType, AddrSpace

from bgpu_assembler import BGPUAssembler
from bgpu_driver import BGPUDriver

import struct
import math
import functools

bgpu_widest_type = dtypes.int
bgpu_addr_type = dtypes.int

bgpu_global_size = 1 << 16
bgpu_local_size = 8
bgpu_max_registers = 256
new_codegen = True
LD_REG_OPTIMIZATION = True
DEAD_CODE_ELIMINATION = True

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

asm_rewrite = PatternMatcher([
  # Range with constant bound
  (UPat(Ops.RANGE, name="x", allow_any_len=True), lambda ctx,x: f"mov.ri.{types[x.dtype]}\t\t{ctx.r[x].rjust(4)}, {"0".rjust(4)} # init range\nloop_{ctx.r[x]}:"),

  # End Range
  (UPat(Ops.END, name="x", src=(UPat.var('last_op'), UPat.var('range'))), lambda ctx,x,last_op,range:
    [f"checkloop_{ctx.r[range]}:",
    f"\tadd.ri.{types[range.dtype]}\t\t{ctx.r[range].rjust(4)}, {ctx.r[range].rjust(4)}, {"1".rjust(4)} # increment",
    f"\tsub.rr.{types[range.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[range].rjust(4)}, {ctx.r[range.src[0]].rjust(4)} # compare bound",
    f"\tbr.nz.loop_{ctx.r[range]}\t\t{ctx.r[x].rjust(4)} # loop back if not done",
    f"endloop_{ctx.r[range]}:"]
  ),

  # Constants -> mov.ri
  (UPat.cvar("x"), lambda ctx, x: f"mov.ri.{types[x.dtype][0:]}\t{ctx.r[x].rjust(4)}, {render_val(x.arg, x.dtype)} # constant"),

  # Load with just a base address-> ld
  (UPat(Ops.LOAD, name="x", src=(UPat.var('base'))),
   lambda ctx, x, base: None \
     if x.dtype.count > 1 else f"ld.{types[x.dtype]}.{mem_type(x)}\t\t{ctx.r[x].rjust(4)}, {ctx.r[base].rjust(4)}"),

  # Gated index -> no-op as it is handled in a gated load
  (UPat(Ops.INDEX, name="x", src=(UPat.var("buf"), UPat.var("loc"), UPat.var("gate"))),
    lambda ctx, x, loc, gate, buf: f"# Gated index {ctx.r[x]}"),

  # Gated Load
  (UPat(Ops.LOAD, name="x", src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("loc"), UPat.var("gate"))), UPat.var("alt"))),
    lambda ctx, x, loc, alt, gate, buf: 
    None if x.dtype.count > 1 else [f"\tmov.rr.{types[x.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[alt].rjust(4)} # load alternative",
    f"\tbr.ez.load_{ctx.r[x]} {ctx.r[gate].rjust(4)} # if gate is zero, skip load",
    f"\tshl.ri.{types[bgpu_addr_type]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[loc].rjust(4)}, {render_val(math.log2(buf.dtype.base.scalar().itemsize), bgpu_addr_type)} # index shift",
    f"\tadd.rr.{types[bgpu_addr_type]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[buf].rjust(4)}, {ctx.r[x].rjust(4)} # index into buffer",
    f"\tld.{types[x.dtype]}.{mem_type(x)}\t\t{ctx.r[x].rjust(4)}, {ctx.r[x].rjust(4)} # load",
    f"load_{ctx.r[x]}: # skip label for load",
    "\tsync.threads"
    ]),

  # Where
  (UPat(Ops.WHERE, name="x", src=(UPat.var('cond'), UPat.var('a'), UPat.var('b'))),
    lambda ctx, x, cond, a, b:
    None if x.dtype.count > 1 else [
      f"\tmov.rr.{types[x.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[b].rjust(4)} # where false case",
      f"\tbr.ez.where_{ctx.r[x]} {ctx.r[cond].rjust(4)} # if cond is zero, skip true case",
      f"\tmov.rr.{types[x.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[a].rjust(4)} # where true case",
      f"where_{ctx.r[x]}: # skip label for where",
      "\tsync.threads"
      ]),

  # Store with a just a base address
  (UPat(Ops.STORE, name="x", src=(UPat.var('base'), UPat.var("var"))), lambda ctx, x, base, var:
    None if var.dtype.count > 1 or x.arg is not None else
    f"st.{types[var.dtype.scalar()]}.{mem_type(base)}\t\t" + \
    f"{ctx.r[base].rjust(4)}, {('{' + ', '.join(ctx.r[var]) + '}') if var.dtype.count > 1 else ctx.r[var].rjust(4)}"),

  # Store register into a register -> mov.rr
  (UPat(Ops.STORE, name="x", src=(UPat.var('base'), UPat.var("var"))), lambda ctx, x, base, var:
    None if var.dtype.count > 1 or x.arg is None or x.arg.op != Ops.DEFINE_REG else
    f"mov.rr.{types[var.dtype]}\t\t\t" + \
    f"{ctx.r[x.arg].rjust(4)}, {ctx.r[var].rjust(4)} # store register into register"),

  # ALU register register
  (UPat({Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE}, name="x", src=(UPat.var('a'), UPat.var('b'))),
   lambda ctx, x, a, b: f"{x.op.name.lower()}.rr.{types[a.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[a].rjust(4)}, {ctx.r[b].rjust(4)}"),

  # ALU register register
  (UPat(GroupOp.ALU, name="x", src=(UPat.var('a'), UPat.var('b'))),
   lambda ctx, x, a, b: f"{x.op.name.lower()}.rr.{types[x.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[a].rjust(4)}, {ctx.r[b].rjust(4)}"),

  # ALU register constant
  (UPat(GroupOp.ALU, name="x", src=(UPat.var('a'))),
   lambda ctx, x, a: None if x.arg == None else f"{x.op.name.lower()}.ri.{types[x.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[a].rjust(4)}, {render_val(x.arg, x.dtype).rjust(4)}"),

  # ALU register
  (UPat(GroupOp.ALU, name="x", src=(UPat.var('a'))),
   lambda ctx, x, a: f"{x.op.name.lower()}.rr.{types[x.dtype]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[a].rjust(4)}"),

  # Parameter
  (UPat(Ops.DEFINE_GLOBAL, name="x"), lambda ctx,x:
    f"ldparam.{types[bgpu_addr_type]} {ctx.r[x].rjust(4)}, {x.arg} # define global"),

  # Index
  (UPat(Ops.INDEX, name="x", src=(UPat.var('a'), UPat.var('b'))),
   lambda ctx, x, a, b: [
    f"\tshl.ri.{types[bgpu_addr_type]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[b].rjust(4)}, {render_val(math.log2(a.dtype.base.scalar().itemsize), bgpu_addr_type)} # index shift",
    f"\tadd.rr.{types[bgpu_addr_type]}\t\t{ctx.r[x].rjust(4)}, {ctx.r[a].rjust(4)}, {ctx.r[x].rjust(4)} # index"
   ]),

  # Special
  (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: 
   f"special\t\t\t{ctx.r[x].rjust(4)}, %{x.arg[0]}"
  ),

  # Cast
  (UPat({Ops.CAST, Ops.BITCAST}, name="x", src=(UPat.var('a'))), lambda ctx,x,a:
    f"cast.{types[x.dtype]}.{types[a.dtype]}\t{ctx.r[x]}, {ctx.r[a]}"
  ),

  # Define Register
  (UPat(Ops.DEFINE_REG, name="x"), lambda ctx, x: f"mov.ri.{types[x.dtype.base.scalar()][0:]}\t{ctx.r[x]}, {x.arg} # define register"),

  # Sink
  (UPat(Ops.SINK), lambda: "stop"),
]
)

class Quadruple:
  def __init__(self, op: str, dst=None, srcs=[], args=[], never_dead=False):
    self.op = op
    self.dst = dst
    self.srcs = srcs
    self.args = args
    self.dead = False # set to true by dead code elimination and then not printed
    self.never_dead = never_dead # for instructions that must not be eliminated (like stores, branches)
    self.pred = []
    self.succ = []
  
  def defs(self):
    return set() if self.dst is None else set([self.dst])

  def uses(self):
    return set(self.srcs)
  
  def __str__(self):
    arg_str = str(self.args)
    if isinstance(self.args, list):
      arg_str = ", ".join(self.args)

    dst_str = "" if self.dst is None else f"{self.dst}, "

    sep_str = ""
    if len(self.srcs) > 0 and arg_str != "":
      sep_str = ", "

    return f"{self.op} {dst_str}{", ".join(self.srcs)}{sep_str}{arg_str}"

class BasicBlock:
  def __init__(self, label: str, insts: list[Quadruple], succ: list[str]):
    self.label = label
    self.insts = insts
    self.succ = succ
    self.pred = []

  def analyze(self):
    # Create edges within the BasicBlock
    for idx, inst in enumerate(self.insts):
      if idx > 0:
        inst.pred = [self.insts[idx-1]]

      if idx < (len(self.insts)-1):
        inst.succ = [self.insts[idx+1]]
  
  def __str__(self):
    res = f"{self.label}: # pred: {" ".join(self.pred)}\n"
    for inst in self.insts:
      if inst.dead:
        assert not inst.never_dead, "Dead code elimination removed an instruction that should not be removed!"
        continue
      res += "\t" + str(inst) + "\n"
    if "stop" in self.succ:
      res += "\tstop\n"
    return res + f"\t# succ: {" ".join(self.succ)}\n"

class Cfg:
  def __init__(self, entry: str, bblocks: list[BasicBlock]):
    self.entry = entry
    self.bblocks = bblocks

  def analyze(self):
    # Helper maps
    self.bblock_map = {}
    self.inst_map = {}
    self.ld_reg_map = {} # ld_reg to alloca map
    uid = 0
    for blk in self.bblocks:
      blk.analyze()
      blk.pred = []
      self.bblock_map[blk.label] = blk
      for inst in blk.insts:
        inst.uid = uid
        self.inst_map[uid] = inst
        uid += 1
        if inst.op == "ld_reg":
          self.ld_reg_map[inst.dst] = inst.srcs[0]

    # Analize block-level control flow
    num_stop_succ = 0
    for blk in self.bblocks:
      if len(blk.succ) < 1 and len(blk.succ) > 2:
        raise RuntimeError(f"BasicBlock {blk.label} has {len(blk.succ)} successors (must be 1 or 2)")
      for succ in blk.succ:
        if succ == "stop":
          num_stop_succ += 1
        else:
          self.bblock_map[succ].pred.append(blk.label)
    
    if num_stop_succ != 1:
      raise RuntimeError(f"Cfg must have exactly one bblock with one 'stop' as successor, found {num_stop_succ}")

    # Determine instruction-level control flow
    for blk in self.bblocks:
      # First instruction has the last instruction of the previous block as pred 
      for pred_blk_name in blk.pred:
        blk.insts[0].pred.append(self.bblock_map[pred_blk_name].insts[-1])

      # Last instruction has the first instruction of the next block as succ
      for succ_blk_name in blk.succ:
        if succ_blk_name != "stop":
          blk.insts[-1].succ.append(self.bblock_map[succ_blk_name].insts[0])

    # Print all instructions
    for uid in self.inst_map:
      inst = self.inst_map[uid]
      print(f"inst {uid}, pred {[i.uid for i in inst.pred]}, succ: {[i.uid for i in inst.succ]}, defs: {inst.defs()}, uses: {inst.uses()}: {str(inst)}")

    # ld_reg optimization: replace uses of ld_reg dst with alloca src
    if LD_REG_OPTIMIZATION:
      for uid in self.inst_map:
        inst = self.inst_map[uid]
        new_srcs = []
        for src in inst.srcs:
          if src in self.ld_reg_map:
            new_srcs.append(self.ld_reg_map[src])
            print(f"Optimizing ld_reg: replacing use of {src} with {self.ld_reg_map[src]}")
          else:
            new_srcs.append(src)
        inst.srcs = new_srcs

    # Liveness analysis
    w = []
    for uid in self.inst_map:
      inst = self.inst_map[uid] 
      inst.in_set = set()
      inst.out_set = set()
      w.append(inst)
    
    while len(w) > 1:
      n = w.pop()
      old_in = n.in_set

      new_out = set()
      for i in n.succ:
        new_out = new_out | i.in_set
      n.out_set = new_out

      new_in = n.uses() | (new_out - n.defs())
      n.in_set = new_in

      if (new_in != old_in):
        for m in n.pred:
          w.append(m)

    # Print liveness
    required_registers = 0
    for uid in self.inst_map:
      inst = self.inst_map[uid]
      required_registers = max(required_registers, len(inst.in_set | inst.out_set))
      if not inst.never_dead and DEAD_CODE_ELIMINATION:
        inst.dead = (inst.defs() & inst.out_set) == set()
      print(f"inst {uid}, dead: {inst.dead}: in: {inst.in_set} out: {inst.out_set}")
    
    print(f"Required registers: {required_registers}")

    if required_registers > bgpu_max_registers:
      raise RuntimeError(f"Kernel requires more than {bgpu_max_registers} registers, spilling not yet implemented!")

    # Linear register allocator
    pal = set([i for i in range(bgpu_max_registers)])
    reg_map = {}

    # Precolor phi nodes -> all should have the same register
    available_regs = list(pal)
    available_regs.sort(reverse=True)
    for uid in self.inst_map:
      inst = self.inst_map[uid]
      if inst.op != "phi":
        continue
      reg = available_regs.pop()
      # Force srcs to use the same register
      for src in inst.srcs:
        reg_map[src] = reg

      # convert to rr move
      inst.op = "mov.rr.int32"
      inst.srcs = [inst.srcs[0]]

    # Color remaining registers
    for uid in self.inst_map:
      inst = self.inst_map[uid]
      live = inst.in_set | inst.out_set
      used = set()
      for reg in live:
        if reg in reg_map:
          used.add(reg_map[reg])
      print(f"inst {uid}: live: {live} used: {used}")
      available_regs = list(pal - used)
      available_regs.sort(reverse=True)
      for d in inst.defs():
        if d not in reg_map:
          reg_map[d] = available_regs.pop()
          print(f"{d} : {reg_map[d]}")

    # Replace registers
    for uid in self.inst_map:
      inst = self.inst_map[uid]
      if inst.dst is not None:
        inst.dst = f"r{reg_map[inst.dst]}"
      new_srcs = []
      for src in inst.srcs:
        new_srcs.append(f"r{reg_map[src]}")
      inst.srcs = new_srcs

    # Replace alloca, ld_reg and st_reg
    # TODO: Proper mem2reg implementation
    for uid in self.inst_map:
      inst = self.inst_map[uid]
      if inst.op == "alloca":
        inst.dead = True # Just reserve the register
      elif inst.op == "ld_reg":
        inst.op = "mov.rr.int32"
      elif inst.op == "st_reg":
        inst.op = "mov.rr.int32"

  def render(self):
    res = ""
    for blk in self.bblocks:
      res += str(blk)
    return res

  def __str__(self):
    return f"CFG, entry: {self.entry}\n" + self.render()

  def from_uops(uops: list[UOp]):
    print("Creating Cfg for:")
    print_uops(uops)

    bblocks = []

    entry_name = "UNDEFINED"
    current_bblock_label = "UNDEFINED"
    current_bblock_insts = []
    uop_to_ssa = {}
    idx = 0
    def ssa(uop=None):
      nonlocal idx
      name = f"%{idx}"
      idx += 1
      if uop is not None:
        uop_to_ssa[uop] = name
      return name

    def get_srcs(uop):
      return [uop_to_ssa[src] for src in uop.src]
      
    alloca_regs = {}
    for uop in uops:
      if uop.op is Ops.DEFINE_GLOBAL:
        current_bblock_insts.append(Quadruple(f"ldparam.{types[bgpu_addr_type]}", ssa(uop), srcs=[], args=uop.arg))
      elif uop.op is Ops.DEFINE_REG:
        for _ in range(uop.dtype.size):
          new_regs = [ssa() for _ in range(uop.dtype.size)]
          for r in new_regs:
            current_bblock_insts.append(Quadruple(f"alloca", r, args=uop.arg))
          alloca_regs[uop] = new_regs
      elif uop.op is Ops.CONST:
        current_bblock_insts.append(Quadruple(f"mov.ri.{types[uop.dtype]}", ssa(uop), args=uop.arg))
      elif uop.op is Ops.SPECIAL:
        current_bblock_insts.append(Quadruple("special", ssa(uop), args=f"%{uop.arg[0]}"))
      elif uop.op in GroupOp.ALU:
        current_bblock_insts.append(Quadruple(f"{uop.op.name.lower()}.rr.{types[uop.dtype]}", ssa(uop), srcs=get_srcs(uop)))
      elif uop.op is Ops.INDEX:
        if uop.dtype.addrspace == AddrSpace.GLOBAL:
          if uop.dtype.base.itemsize > 1:
            shift_name = ssa(uop)
            shift_val = int(math.log2(uop.src[0].dtype.base.itemsize))
            current_bblock_insts.append(Quadruple(f"shl.ri.{types[bgpu_addr_type]}", shift_name, srcs=[uop_to_ssa[uop.src[1]]], args=str(shift_val)))
            current_bblock_insts.append(Quadruple(f"add.rr.{types[bgpu_addr_type]}", shift_name, srcs=[uop_to_ssa[uop.src[0]], shift_name]))
          else:
            current_bblock_insts.append(Quadruple(f"add.rr.{types[bgpu_addr_type]}", ssa(uop), srcs=get_srcs(uop)))
        elif uop.dtype.addrspace == AddrSpace.REG:
          assert uop.src[1].op == Ops.CONST
          print(f"index into register {uop.src[0]} at constant {uop.src[1].arg}")
          uop_to_ssa[uop] = alloca_regs[uop.src[0]][uop.src[1].arg]
        else:
          raise NotImplementedError(f"Index not implemented for {uop.dtype.addrspace}")
      elif uop.op is Ops.LOAD:
        if uop.src[0].dtype.addrspace == AddrSpace.GLOBAL:
          current_bblock_insts.append(Quadruple(f"ld.{types[uop.dtype]}.global", ssa(uop), srcs=get_srcs(uop)))
        elif uop.src[0].dtype.addrspace == AddrSpace.REG:
          current_bblock_insts.append(Quadruple(f"ld_reg", ssa(uop), srcs=get_srcs(uop)))
        else:
          raise NotImplementedError(f"Load not implemented for {uop.src[0].dtype.addrspace}")
      elif uop.op is Ops.STORE:
        if uop.src[0].dtype.addrspace == AddrSpace.REG:
          current_bblock_insts.append(Quadruple(f"st_reg", None, srcs=get_srcs(uop), never_dead=True))
        else:
          current_bblock_insts.append(Quadruple(f"st.{types[uop.src[0].dtype.base]}.global", None, srcs=get_srcs(uop), never_dead=True))
      elif uop.op is Ops.RANGE:
        if uop.src[0].arg <= 0:
          raise RuntimeError(f"Range has 0 or fewer iterations: {uop.src[0].arg}")
        range_name = ssa(uop)
        loop_entry_name = f"loop_entry_{range_name}"
        loop_check_name = f"loop_check_{range_name}"
        loop_body_name = f"loop_body_{range_name}"
        loop_exit_name = f"loop_exit_{range_name}"
        # close basic block -> succ is loop_entry
        bblocks.append(
          BasicBlock(current_bblock_label, current_bblock_insts, [loop_entry_name])
        )
        # loop entry basic block -> succ is loop_check
        # initialize counter
        counter_init_name = f"%{loop_entry_name}_init"
        bblocks.append(
          BasicBlock(loop_entry_name, [Quadruple(f"mov.ri.{types[uop.dtype]}", counter_init_name, args="0")], [loop_check_name])
        )
        # loop check basic block
        bblocks.append(
          BasicBlock(loop_check_name, [
            Quadruple("phi", range_name, srcs=[f"{range_name}_phi", counter_init_name]),
            Quadruple(f"add.ri.{types[uop.dtype]}", f"{range_name}_phi", srcs=[range_name], args="1"),
            Quadruple(f"sub.ri.{types[uop.dtype]}", f"{range_name}_cmp", srcs=[f"{range_name}_phi"], args=uop.src[0].arg+1),
            Quadruple(f"br.ez.{loop_exit_name}", srcs=[f"{range_name}_cmp"], never_dead=True)
          ],
          [loop_body_name, loop_exit_name])
        )
        # We are now in the loop_body_name
        current_bblock_label = loop_body_name
        current_bblock_insts = []
      elif uop.op is Ops.AFTER:
        alloca_regs[uop] = alloca_regs[uop.src[0]] # map to alloca'd regs
      elif uop.op is Ops.END:
        # close basic block
        range_name = uop_to_ssa[uop.src[1]]
        loop_check_name = f"loop_check_{range_name}"
        loop_footer_name = f"loop_footer_{range_name}"
        loop_exit_name = f"loop_exit_{range_name}"
        # close basic block -> succ is loop_footer
        bblocks.append(
          BasicBlock(current_bblock_label, current_bblock_insts, [loop_footer_name])
        )
        # loop footer jumps to loop check
        bblocks.append(
          BasicBlock(loop_footer_name, [Quadruple(f"br.nz.{loop_check_name}", srcs=[f"{range_name}_cmp"], never_dead=True)], [loop_check_name])
        )
        # start loop exit block
        current_bblock_label = loop_exit_name
        current_bblock_insts = []
      elif uop.op is Ops.SINK:
        entry_name = uop.arg.name
        if len(bblocks) == 0:
          current_bblock_label = entry_name
        else:
          bblocks[0].label = entry_name
      elif uop.op is Ops.GROUP:
        continue
      else:
        print("Previous blocks:")
        for b in bblocks:
          print(b.__str__())

        print("Current block:")
        print(f"{current_bblock_label}:")
        for inst in current_bblock_insts:
          print(f"\t{inst}")
        raise NotImplementedError(f"Uop {uop.op} not implemented!")

    bblocks.append(
      BasicBlock(current_bblock_label, current_bblock_insts, ["stop"])
    )

    return Cfg(entry_name, bblocks)

class BGPURenderer(Renderer):
  device = "BGPU"
  suffix = ".bgpu"
  supports_float4 = False
  has_local = True
  has_threads = False
  has_shared = False
  global_max = (bgpu_global_size, 1, 1)
  local_max = (bgpu_local_size, 1, 1)
  shared_max = 0
  tensor_cores = []
  pre_matcher = None
  extra_matcher = None
  code_for_op = asm_for_op

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

    cfg = Cfg.from_uops(uops)
    print(cfg)
    cfg.analyze()
    print(cfg)

    if new_codegen:
      return cfg.render()

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
      prefix += f"_{dtype if dtype is not None else types[cast(UOp, u).dtype]}_"
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
      if u.op in [Ops.NOOP, Ops.GROUP]:
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
        r[u] = [ssa('val', dtype=types[u.dtype.scalar()]) for _ in range(u.dtype.count)] if u.dtype.count > 1 else ssa('val', u)
        continue
      elif u.op is Ops.DEFINE_GLOBAL: 
        print(f"global {u.arg} of type {u.dtype}")
        bufs.append((f"data{u.arg}", u.dtype))
      
      prefix, dtype = {
          Ops.END: ("range_cond", types[dtypes.int]),
          Ops.RANGE: ("range", types[u.dtype.base.scalar()]),
          Ops.INDEX: ("idx", types[u.dtype.base.scalar()]),
          Ops.CAST: ("cast", None),
          Ops.BITCAST: ("cast", None),
          Ops.CONST: ("const", None),
          Ops.DEFINE_GLOBAL: ("param", types[bgpu_widest_type]),
          Ops.DEFINE_REG: ("reg", types[u.dtype.base.scalar()]),
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

      if not has_range and False:
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
    super().__init__(device, BGPUAllocator(self), CompilerSet([CompilerPair(BGPURenderer, Compiler)]), functools.partial(BGPUProgram, self))
