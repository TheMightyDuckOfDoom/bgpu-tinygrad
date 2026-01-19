from typing import Callable
from tinygrad.device import Compiled, LRUAllocator, BufferSpec, Compiler, CompilerSet, CompilerPair
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import UOp, Ops, GroupOp, print_uops
from tinygrad.dtype import dtypes, DType, AddrSpace

from bgpu_assembler import BGPUAssembler
from bgpu_driver import BGPUDriver

import math
import functools

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

  def render(self, uops:list[UOp]) -> str:
    print("Rendering BGPU code")

    cfg = Cfg.from_uops(uops)
    print(cfg)
    cfg.analyze()
    print(cfg)

    return cfg.render()

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
