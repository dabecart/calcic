"""
TAC_optimizer.py

Applies optimization techniques to reduce the space of the program and improve speed by applying:
- Constant folding.
- Unreachable code elimination.
- Copy propagation.
- Dead store elimination.

calcic. Written by @dabecart, 2026.
"""

from __future__ import annotations
from .TAC import *
from .types import *
from dataclasses import dataclass

@dataclass
class TACOptimizationFlags:
    constant_folding: bool
    unreachable_code_elimination: bool
    copy_propagation: bool
    dead_store_elimination: bool

class TACOptimizer:
    def __init__(self, optimizationFlags: TACOptimizationFlags) -> None:
        self.flags = optimizationFlags
    
    def optimize(self, tacProgram: TACProgram):
        optimizedTAC: list[TACTopLevel] = []
        for top in tacProgram.topLevel:
            # Optimizations are carried out at function level.
            if isinstance(top, TACFunction):
                top.instructions = self._optimizeFunction(top.instructions)
            optimizedTAC.append(top)

        # Set the program top level with the optimized blocks.
        tacProgram.topLevel = optimizedTAC

    def _optimizeFunction(self, funcInsts: list[TACInstruction]) -> list[TACInstruction]:
        if len(funcInsts) == 0:
            return funcInsts
        
        while True:
            if self.flags.constant_folding:
                optInsts = self.runConstantFolding(funcInsts)
            else:
                optInsts = funcInsts

            # cfg = ControlFlowGraph(optInsts)

            # if self.flags.unreachable_code_elimination:
            #     cfg = self.runUnreachableCodeElimination(cfg)
            # if self.flags.copy_propagation:
            #     cfg = self.runCopyPropagation(cfg)
            # if self.flags.dead_store_elimination:
            #     cfg = self.runDeadStoreElimination(cfg)

            # optInsts = cfg.toInstructionList()

            if optInsts == funcInsts or len(optInsts) == 0:
                return optInsts
            
            # Continue the loop.
            funcInsts = optInsts
    
    def runConstantFolding(self, insts: list[TACInstruction]) -> list[TACInstruction]:
        optInsts: list[TACInstruction] = []

        for inst in insts:
            foldable: bool = False
            warning: StaticEvalMsg = EVAL_OK
            match inst:
                case TACUnary():
                    if foldable := inst.exp.isConstant:
                        # Can be folded, evaluate and convert into a TACCopy instruction.
                        value, warning = StaticEvaluation.evalDecl(
                                            inst.operator.value, 
                                            inst.exp.valueType, inst.result.valueType,
                                            inst.exp.constantValue)
                        # Instruction can be folded, replace by a TACCopy instruction.
                        foldedConst = TACValue(True, inst.result.valueType, str(value))
                        # This adds itself to optInsts.
                        TACCopy(foldedConst, inst.result, optInsts)
                
                case TACBinary():
                    if foldable := (inst.exp1.isConstant and inst.exp2.isConstant):
                        value, warning = StaticEvaluation.evalDecl(
                                            inst.operator.getBaseOperator().value, 
                                            inst.exp1.valueType, inst.result.valueType,
                                            inst.exp1.constantValue, inst.exp2.constantValue)
                        foldedConst = TACValue(True, inst.result.valueType, str(value))
                        TACCopy(foldedConst, inst.result, optInsts)

                case TACJumpIfZero():
                    if foldable := inst.condition.isConstant:
                        # If the condition is met, replace with a simpler TACJump. If the condition 
                        # isn't met, remove the instruction.
                        if inst.condition.valueType.isDecimal():
                            if float(inst.condition.constantValue) == 0.0:
                                TACJump(inst.target, optInsts)
                        else:
                            if int(inst.condition.constantValue) == 0:
                                TACJump(inst.target, optInsts)

                case TACJumpIfNotZero():
                    if foldable := inst.condition.isConstant:
                        # If the condition is met, replace with a simpler TACJump. If the condition 
                        # isn't met, remove the instruction.
                        if inst.condition.valueType.isDecimal():
                            if float(inst.condition.constantValue) != 0.0:
                                TACJump(inst.target, optInsts)
                        else:
                            if int(inst.condition.constantValue) != 0:
                                TACJump(inst.target, optInsts)

                case TACJumpIfValue():
                    if foldable := (inst.condition.isConstant and inst.value.isConstant):
                        # If the condition is met, replace with a simpler TACJump. If the condition 
                        # isn't met, remove the instruction.
                        if inst.condition.valueType.isDecimal():
                            if float(inst.condition.constantValue) == float(inst.value.constantValue):
                                TACJump(inst.target, optInsts)
                        else:
                            if int(inst.condition.constantValue) == int(inst.value.constantValue):
                                TACJump(inst.target, optInsts)

                case TACTruncate() | TACSignExtend() | TACZeroExtend() | \
                    TACDecimalToInt()| TACDecimalToUInt()| TACIntToDecimal()| TACUIntToDecimal():
                    if foldable := inst.exp.isConstant:
                        if isinstance(inst.result.valueType, BaseDeclaratorType):
                            toType = inst.result.valueType.baseType
                        else:
                            # TODO: This is only for 64 bits.
                            toType = TypeSpecifier.ULONG

                        value, warning = StaticEvaluation.parseValue(toType, inst.exp.constantValue)
                        foldedConst = TACValue(True, inst.result.valueType, str(value))
                        TACCopy(foldedConst, inst.result, optInsts)

            if warning:
                warning.rise(print, print)

            if not foldable:
                # Instruction cannot be folded, reintroduce in the optInsts list as it is.
                optInsts.append(inst)

        return optInsts
    
    def runUnreachableCodeElimination(self, cfg: ControlFlowGraph) -> ControlFlowGraph:
        return cfg

    def runCopyPropagation(self, cfg: ControlFlowGraph) -> ControlFlowGraph:
        return cfg

    def runDeadStoreElimination(self, cfg: ControlFlowGraph) -> ControlFlowGraph:
        return cfg
        
class ControlFlowGraph:
    def __init__(self, insts: list[TACInstruction]) -> None:
        pass

    def toInstructionList(self) -> list[TACInstruction]:
        return []