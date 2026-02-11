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

    # Aliased variables are those whose address is used during the program. This type of alias 
    # analysis is called "Address Taken Analysis".
    def _getAliasedVariables(self, insts: list[TACInstruction]) -> set[TACValue]:
        # We'll suppose that static variables are also aliased variables, as their address can be 
        # get from other functions in the code.
        aliased: set[TACValue] = set([
            TACValue(False, staticVbe.valueType, staticVbe.identifier)
            for staticVbe in TACStaticVariable.staticVariables.values()
        ])
        for inst in insts:
            if isinstance(inst, TACGetAddress):
                aliased.add(inst.src)
        return aliased
    
    def _optimizeFunction(self, funcInsts: list[TACInstruction]) -> list[TACInstruction]:
        if len(funcInsts) == 0:
            return funcInsts
        
        while True:
            aliasedVariables = self._getAliasedVariables(funcInsts)

            if self.flags.constant_folding:
                optInsts = self.runConstantFolding(funcInsts)
            else:
                optInsts = funcInsts

            cfg = ControlFlowGraph(optInsts)

            if self.flags.unreachable_code_elimination:
                cfg = self.runUnreachableCodeElimination(cfg)
            if self.flags.copy_propagation:
                cfg = self.runCopyPropagation(cfg, aliasedVariables)
            if self.flags.dead_store_elimination:
                cfg = self.runDeadStoreElimination(cfg, aliasedVariables)

            optInsts = cfg.toInstructionList()

            # If there are no more instructions, exit.
            if len(optInsts) == 0:
                return optInsts
            
            # Are the same instructions?
            if len(optInsts) == len(funcInsts) and all(ins1 == ins2 for ins1, ins2 in zip(optInsts, funcInsts)):
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
                            # TODO: This is only for 64-bit systems.
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
        # >> Delete unreachable nodes. 
        # For that, traverse the graph keeping track of the nodes nodes visited.
        visitedNodes = set(cfg.traverse())
        allNodes = set(cfg.nodes.keys())

        # After that, remove those that weren't visited.
        toRemoveNodes = allNodes - visitedNodes
        if len(toRemoveNodes) > 0:
            for id in toRemoveNodes:
                del cfg.nodes[id]

            # Delete the references to the deleted nodes.
            for nodeID in cfg.nodes.values():
                nodeID.predecessors -= toRemoveNodes
                nodeID.successors -= toRemoveNodes

        # >> Delete useless labels and jumps. 
        # Sort the current nodes in a list based on their ID.
        sortedNodeIDs: list[int] = sorted(cfg.nodes.keys())
        for i, nodeID in enumerate(sortedNodeIDs):
            node = cfg.nodes[nodeID]
            # Find nodes with a starting label.
            if isinstance(node.getInitiator(), TACLabel):
                # Get the previous node.
                if i == 0:
                    prevNodeId = cfg.entryNode.nodeID
                else:
                    prevNodeId = sortedNodeIDs[i - 1]
                # Check the predecessors of the current node. If it only has prevNode as 
                # predecessor, the label can be removed.
                if node.predecessors == {prevNodeId}:
                    node.instructions.pop(0)
            
            # Find nodes with a jump terminator.
            if isinstance(node.getTerminator(), (TACJump, TACJumpIfZero, TACJumpIfNotZero, TACJumpIfValue)):
                # Get the next node to which the execution of the program will normally fall into.
                if i == len(sortedNodeIDs) - 1:
                    nextNodeId = cfg.exitNode.nodeID
                else:
                    nextNodeId = sortedNodeIDs[i + 1]
                # Check the successors of the current node. If it only has nextNode as successor, 
                # the jump instruction at the end of the current node can be removed.
                if node.successors == {nextNodeId}:
                    node.instructions.pop()

        # >> Remove empty nodes.
        toRemoveNodes: set[int] = set(
            [n.nodeID for n in cfg.nodes.values() if len(n.instructions) == 0]
        )
        # Remove from the dictionary.
        if len(toRemoveNodes) > 0:
            for id in toRemoveNodes:
                del cfg.nodes[id]
            # Remove from antecessor/successors.
            for node in cfg.nodes.values():
                node.predecessors -= toRemoveNodes
                node.successors -= toRemoveNodes

        return cfg

    def runCopyPropagation(self, cfg: ControlFlowGraph, aliased: set[TACValue]) -> ControlFlowGraph:
        # This will calculate the reaching copies of all instructions.
        cfg.calculateReachingCopies(aliased)
        # Apply the reaching copies of all nodes.
        for node in cfg.nodes.values():
            node.applyReachingCopies()
        return cfg

    def runDeadStoreElimination(self, cfg: ControlFlowGraph, aliased: set[TACValue]) -> ControlFlowGraph:
        # This will calculate the live variables of all instructions.
        cfg.calculateLiveVariables(aliased)
        # Apply the live variables of all nodes.
        for node in cfg.nodes.values():
            node.applyLiveVariables()
        return cfg


"""
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
CONTROL FLOW NODE
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""
@dataclass(eq=False)
class ControlFlowNode:
    nodeID: int
    instructions: list[TACInstruction]      = field(default_factory=list)
    predecessors: set[int]                  = field(default_factory=set)
    successors: set[int]                    = field(default_factory=set)

    # For the copy propagation algorithm:
    # Stores the reaching TACCopy instructions at the end of the node's instructions.
    reachingCopies: set[TACCopy]            = field(default_factory=set)
    # For the dead store elimination algorithm:
    # Stores the TACValues that are alive at the start of the node's instructions.
    liveVariables: set[TACValue]            = field(default_factory=set)

    def connectSuccessor(self, successor: ControlFlowNode):
        self.successors.add(successor.nodeID)
        successor.predecessors.add(self.nodeID)

    def __hash__(self) -> int:
        return hash(self.nodeID)

    # Returns the first instruction, which is usually a label.
    def getInitiator(self) -> TACInstruction|None:
        return self.instructions[0] if len(self.instructions) > 0 else None

    # Returns the last instruction, which usually dictates the jump.
    def getTerminator(self) -> TACInstruction|None:
        return self.instructions[-1] if len(self.instructions) > 0 else None

    """
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    COPY PROPAGATION
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    """
    # Iterates over each instruction and anotates each one with the TACCopy instructions that reach
    # it. This will be used to run the Copy Propagation algorithm. 
    # - startingCopies receives the reaching copy instructions from a predecessor node.
    # - aliased contains the variables whose address is calculated at some point in the code.
    # At the end, the block gets anotated with the reaching copies of the last instruction.
    # Note: This function is also called the transfer function in Data Flow Analysis.
    def anotateReachingCopies(self, startingCopies: set[TACCopy], aliased: set[TACValue]):
        reachingCopies = startingCopies

        for inst in self.instructions:
            # Set the reaching copies for this instruction.
            inst.reachingCopies = set(reachingCopies)

            match inst:
                case TACCopy():
                    # If we have x = y, and this instruction is y = x, we are left with the same 
                    # thing.
                    if TACCopy(inst.dst, inst.src, []) in reachingCopies:
                        continue

                    # Search for copies inside reachingCopies which get modified (or killed) by this 
                    # instruction.
                    copiesToRemove: set[TACCopy] = set()
                    for copy in reachingCopies:
                        if inst.dst in (copy.src, copy.dst):
                            copiesToRemove.add(copy)
                    reachingCopies -= copiesToRemove

                    # Add the current copy to the reachingCopies set only if its operands are the 
                    # same type or if they are both pointers. If they're not, then this copy 
                    # instruction is to transfer data between types and this would create two 
                    # separate variables.
                    # char and signed char are different types in the parser, but not here.
                    sameTypes = inst.src.valueType == inst.dst.valueType
                    bothPointers = isinstance(inst.src.valueType, PointerDeclaratorType) and isinstance(inst.dst.valueType, PointerDeclaratorType)
                    charTuple = (TypeSpecifier.CHAR.toBaseType(), TypeSpecifier.SIGNED_CHAR.toBaseType())
                    bothChars = inst.src.valueType in charTuple and inst.dst.valueType in charTuple
                    if sameTypes or bothPointers or bothChars:
                        reachingCopies.add(inst)
                
                case TACFunctionCall():
                    # Instead of analyzing the behavior of the function being called and how it 
                    # affects the reaching copies of static variables, whenever a function call is 
                    # made, all reaching copies related with static variables will get killed.
                    # Take also into account the returning value of the function call: if it is 
                    # modified, it also gets killed.
                    copiesToRemove: set[TACCopy] = set()
                    for copy in reachingCopies:
                        if copy.src in aliased or copy.dst in aliased or \
                           inst.result in (copy.src, copy.dst):
                            copiesToRemove.add(copy)
                    reachingCopies -= copiesToRemove

                case TACStore():
                    # We are going to suppose that a TACStore updates all aliased variables (this 
                    # is a very conservative approach). Therefore, all copies with aliased variables
                    # will get killed.
                    copiesToRemove: set[TACCopy] = set()
                    for copy in reachingCopies:
                        if copy.src in aliased or copy.dst in aliased:
                            copiesToRemove.add(copy)
                    reachingCopies -= copiesToRemove

                case TACUnary() | TACBinary() | TACSignExtend() | TACZeroExtend() | \
                     TACDecimalToDecimal() | TACDecimalToInt() | TACDecimalToUInt() | \
                     TACIntToDecimal() | TACUIntToDecimal() | TACTruncate() | \
                     TACGetAddress() | TACLoad() | TACStore() | TACAddToPointer() | \
                     TACCopyToOffset() | TACCopyFromOffset():
                    # These instructions only kill copies if their result modify one either the 
                    # source or destination of a reaching copy.
                    copiesToRemove: set[TACCopy] = set()
                    for copy in reachingCopies:
                        if inst.result in (copy.src, copy.dst):
                            copiesToRemove.add(copy)
                    reachingCopies -= copiesToRemove

                case TACReturn() | TACLabel() | \
                     TACJump() | TACJumpIfZero() | TACJumpIfNotZero() | TACJumpIfValue():
                    continue

                case _:
                    raise ValueError(f"Cannot anotate reaching copies of a {type(inst)} instruction")
        
        self.reachingCopies = reachingCopies

    def _replaceOperand(self, operand: TACValue, instReachingCopies: set[TACCopy]) -> tuple[bool, TACValue]:
        if operand.isConstant:
            return (False, operand)
        
        # E.g. if we reach z = y + 10, and before that we had y = x (Copy x to y), we can replace y 
        # by x.
        for copy in instReachingCopies:
            # Don't substitute something with the same thing! It creates an infinite loop!
            if copy.dst == operand and copy.src != operand:
                return (True, copy.src)
        
        return (False, operand)
    
    def _rewriteInstWithReachingCopies(self, inst: TACInstruction) -> TACInstruction|None:
        # If the instruction is not replaceable, the same instruction will be returned at the end of 
        # this function.
        match inst:
            case TACCopy():
                for copy in inst.reachingCopies:
                    # We can cut this TACCopy instruction if the same copy instruction exists or if
                    # x = y reaches the instruction y = x.
                    if (copy == inst) or (copy.src == inst.dst and copy.dst == inst.src):
                        return None
                    
                isReplaceable, newSrc = self._replaceOperand(inst.src, inst.reachingCopies)
                if isReplaceable:
                    return TACCopy(newSrc, inst.dst, [])
            
            case TACUnary():
                isReplaceable, newExp = self._replaceOperand(inst.exp, inst.reachingCopies)
                if isReplaceable:
                    ret = TACUnary(inst.operator, newExp, [])
                    ret.result = inst.result
                    return ret
            
            case TACBinary():
                isReplaceable1, newExp1 = self._replaceOperand(inst.exp1, inst.reachingCopies)
                isReplaceable2, newExp2 = self._replaceOperand(inst.exp2, inst.reachingCopies)
                if isReplaceable1 or isReplaceable2:
                    ret = TACBinary(inst.operator, newExp1, newExp2, instructionsList=[])
                    ret.result = inst.result
                    return ret

            case TACReturn():
                isReplaceable, newResult = self._replaceOperand(inst.result, inst.reachingCopies)
                if isReplaceable:
                    ret = TACReturn(newResult, [])
                    return ret
                
            case TACFunctionCall():
                isReplaceable: bool = False
                newArguments: list[TACValue] = []
                for arg in inst.arguments:
                    isArgumentReplaceable, newArgument = self._replaceOperand(arg, inst.reachingCopies)
                    
                    isReplaceable = isReplaceable or isArgumentReplaceable
                    newArguments.append(newArgument)
                
                if isReplaceable:
                    ret = TACFunctionCall(inst.identifier, inst.returnType, newArguments, [])
                    ret.result = inst.result
                    return ret
                
            case TACJumpIfZero() | TACJumpIfNotZero():
                isReplaceable, newCondition = self._replaceOperand(inst.condition, inst.reachingCopies)
                if isReplaceable:
                    ret = type(inst)(newCondition, inst.target, [])
                    return ret

            case TACJumpIfValue():
                isConditionReplaceable, newCondition = self._replaceOperand(inst.condition, inst.reachingCopies)
                isValueReplaceable, newValue = self._replaceOperand(inst.value, inst.reachingCopies)
                if isConditionReplaceable or isValueReplaceable:
                    ret = TACJumpIfValue(newCondition, newValue, inst.target, [])
                    return ret

            case TACSignExtend() | TACZeroExtend() |  TACTruncate() | \
                 TACDecimalToDecimal() | TACDecimalToInt() | TACDecimalToUInt() | \
                 TACIntToDecimal() | TACUIntToDecimal():
                isReplaceable, newExp = self._replaceOperand(inst.exp, inst.reachingCopies)
                if isReplaceable:
                    ret = type(inst)(newExp, inst.castType, [])
                    ret.result = inst.result
                    return ret
            
            case TACLoad() | TACStore():
                isReplaceable, newSrc = self._replaceOperand(inst.src, inst.reachingCopies)
                if isReplaceable:
                    return type(inst)(newSrc, inst.dst, [])
                
            case TACAddToPointer():
                isPointerReplaceable, newPointer = self._replaceOperand(inst.pointer, inst.reachingCopies)
                isIndexReplaceable, newIndex = self._replaceOperand(inst.index, inst.reachingCopies)
                if isPointerReplaceable or isIndexReplaceable:
                    return TACAddToPointer(newPointer, newIndex, inst.scale, inst.dst, [])

            case TACCopyToOffset():
                isReplaceable, newSrc = self._replaceOperand(inst.src, inst.reachingCopies)
                if isReplaceable:
                    return TACCopyToOffset(newSrc, inst.dst, inst.byteOffset, [])

            case TACCopyFromOffset():
                isReplaceable, newSrc = self._replaceOperand(inst.src, inst.reachingCopies)
                if isReplaceable:
                    return TACCopyFromOffset(newSrc, inst.byteOffset, inst.dst, [])

            case TACJump() | TACLabel() | TACGetAddress():
                # TACGetAddress uses the address of the source and not the value; therefore, the
                # source cannot be substituted.
                pass

            case _:
                raise ValueError(f"Invalid instruction {type(inst)}")

        return inst

    def applyReachingCopies(self):
        newInstructionList: list[TACInstruction] = []
        for inst in self.instructions:
            newInst = self._rewriteInstWithReachingCopies(inst)
            if newInst is not None:
                newInstructionList.append(newInst)
        
        self.instructions = newInstructionList

    """
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    DEAD STORE ELIMINATION
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    """
    # Starting from the latest instruction in the node, annotate which variables are alive just
    # before each instruction. 
    def anotateLiveVariables(self, endingVariables: set[TACValue], aliasedVariables: set[TACValue]):
        liveVariables = endingVariables

        for inst in reversed(self.instructions):
            inst.liveVariables = set(liveVariables)

            match inst:
                case TACReturn():
                    if not inst.returnValue.isConstant:
                        liveVariables.add(inst.returnValue)

                case TACCopy():
                    if inst.dst in liveVariables:
                        liveVariables.remove(inst.dst)

                    if not inst.src.isConstant:
                        liveVariables.add(inst.src)
                
                case TACBinary():
                    # The result is either a new variable or a modification of a previous one. In 
                    # either case, this instruction doesn't care for the value it had, therefore, 
                    # it is killed.
                    if inst.result in liveVariables:
                        liveVariables.remove(inst.result)

                    # The terms in the instruction need to be declared somewhere previously in the 
                    # code, if they are variables. They need to be alive until the execution of the 
                    # instruction.
                    if not inst.exp1.isConstant:
                        liveVariables.add(inst.exp1)
                    if not inst.exp2.isConstant:
                        liveVariables.add(inst.exp2)

                case TACUnary():
                    if inst.result in liveVariables:
                        liveVariables.remove(inst.result)

                    if not inst.exp.isConstant:
                        liveVariables.add(inst.exp)

                case TACJumpIfZero() | TACJumpIfNotZero():
                    if not inst.condition.isConstant:
                        liveVariables.add(inst.condition)

                case TACJumpIfValue():
                    if not inst.condition.isConstant:
                        liveVariables.add(inst.condition)
                    if not inst.value.isConstant:
                        liveVariables.add(inst.value)

                case TACFunctionCall():
                    if inst.result in liveVariables:
                        liveVariables.remove(inst.result)

                    for arg in inst.arguments:
                        if not arg.isConstant:
                            liveVariables.add(arg)

                    # We'll suppose that all aliased variables will be live before a function call,
                    # as they may be needed inside.
                    liveVariables |= aliasedVariables

                case TACLabel() | TACJump():
                    continue

                case _:
                    raise ValueError(f"Cannot anotate live variables of a {type(inst)} instruction")
        
        self.liveVariables = liveVariables

    def _isDeadStore(self, inst: TACInstruction) -> bool:
        match inst:
            case TACFunctionCall():
                # We cannot eliminate function calls as they may affect other parts of the code.
                return False
            
            case TACUnary() | TACBinary() | TACCopy():
                if inst.result not in inst.liveVariables:
                    return True
                
        return False

    def applyLiveVariables(self):
        # If an instruction is a dead store, remove it.
        newInstructions = [
            inst
            for inst in self.instructions
            if not self._isDeadStore(inst)
        ]
        self.instructions = newInstructions

"""
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
CONTROL FLOW GRAPH
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""
class ControlFlowGraph:
    def __init__(self, insts: list[TACInstruction]) -> None:
        self.entryNode = ControlFlowNode(-1)
        self.exitNode  = ControlFlowNode(-2)
        
        self.nodes, self.names = self._createNodes(insts)
        self._connectAllNodes()

    def toInstructionList(self) -> list[TACInstruction]:
        ret: list[TACInstruction] = []
        for nodeID in sorted(self.nodes.keys()):
            ret.extend(self.nodes[nodeID].instructions)
        return ret
    
    # Traverse the graph in using Depth First Search (DFS).
    # Returns a list of all the visited node's IDs in postorder (the children are added before the 
    # parent).
    #       A
    #     /   \   
    #    B     C        -> [D, B, C, A]
    #     \   /
    #       D
    def traverse(self) -> list[int]:
        # Use the set to keep track of visited nodes, for O(1) search.
        visitedNodesSet: set[int] = set()
        # Use the list to keep track of the order.
        orderedNodes: list[int] = []

        def postorder(nodeId: int):
            # The node is now visited.
            visitedNodesSet.add(nodeId)
            # Order the successors.
            for succ in self._getNodeByID(nodeId).successors:
                if succ not in visitedNodesSet:
                    postorder(succ)
            # When the successors have been added, add the current node.
            orderedNodes.append(nodeId)

        postorder(self.entryNode.nodeID)
        return orderedNodes

    def _createNodes(self, insts: list[TACInstruction]) -> tuple[dict[int, ControlFlowNode], dict[str, int]]:
        # Key: ID of the node. Value: the node.
        nodes: dict[int, ControlFlowNode] = {}
        # Key: the name of the node (the top label). Value: the node.
        names: dict[str, int] = {}

        currentNode = ControlFlowNode(0)
        for inst in insts:
            if isinstance(inst, TACLabel):
                # If there's a current node, it is being closed by this label. Time to create a new
                # one, whose first instruction will be the TACLabel.
                if len(currentNode.instructions) > 0:
                    nodes[len(nodes)] = currentNode
                
                currentNode = ControlFlowNode(len(nodes))
                currentNode.instructions.append(inst)

                # Associate label name with the new node's ID.
                names[inst.identifier] = currentNode.nodeID
            
            elif isinstance(inst, (TACJump, TACJumpIfZero, TACJumpIfNotZero, TACJumpIfValue, TACReturn)):
                # This is the last instruction of the current node.
                currentNode.instructions.append(inst)
                nodes[len(nodes)] = currentNode

                currentNode = ControlFlowNode(len(nodes))

            else:
                currentNode.instructions.append(inst)

        # Close the final node.
        if len(currentNode.instructions) > 0:
            nodes[len(nodes)] = currentNode

        return (nodes, names)
    
    # Connects the nodes by setting their predecessor and successors.
    def _connectAllNodes(self):
        # First, connect the entry node with the first node of the list.
        self.entryNode.connectSuccessor(self.nodes[0])

        # Then, connect all nodes.
        for id, node in self.nodes.items():
            if id == len(self.nodes) - 1:
                nextNode = self.exitNode
            else:
                nextNode = self.nodes[id + 1]

            # Depending on the latest instruction in the node...
            finalInst = node.getTerminator()
            match finalInst:
                case TACReturn():
                    node.connectSuccessor(self.exitNode)
                case TACJump():
                    # Connect the target node with the current one. Use the name of the tag to do 
                    # so.
                    targetNode = self.nodes[self.names[finalInst.target]]
                    node.connectSuccessor(targetNode)
                case TACJumpIfZero() | TACJumpIfNotZero() | TACJumpIfValue():
                    # Connect to the target node and the next one, the target will be reached in 
                    # case the condition is met, and the next node otherwise.
                    targetNode = self.nodes[self.names[finalInst.target]]
                    node.connectSuccessor(targetNode)
                    node.connectSuccessor(nextNode)
                case _:
                    node.connectSuccessor(nextNode)

    def _getNodeByID(self, id: int) -> ControlFlowNode:
        if id == -1:
            return self.entryNode
        if id == -2:
            return self.exitNode
        
        if id not in self.nodes:
            raise ValueError(f"Node with id {id} does not exist")

        return self.nodes[id]

    def _getNodeByName(self, name: str) -> ControlFlowNode:
        nodeID = self.names.get(name)
        if nodeID is None:
            raise ValueError(f"Node named {name} does not exist")
        return self._getNodeByID(nodeID)
    
    def _getAllCopyInstructions(self) -> list[TACCopy]:
        foundCopies: list[TACCopy] = []
        for node in self.nodes.values():
            foundInNode = [ins for ins in node.instructions if isinstance(ins, TACCopy)]
            foundCopies.extend(foundInNode)
        return foundCopies

    """
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    COPY PROPAGATION
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    """
    # Calculates the input reaching copies of the node, that is, the intersection of the outgoing 
    # reaching copies of the predecessor nodes. If all predecessors have the same reaching copy, 
    # then it reaches this node.
    # This is also called the meet operator in Data Flow Theory.
    def _calculateReachingCopiesToNode(self, node: ControlFlowNode) -> set[TACCopy]:
        predecessorList = [self._getNodeByID(id) for id in list(node.predecessors)]
        
        # No predecessors, no reaching copies. This block would be removed by the unreachable code
        # algorithm, but this is left as a safeguard.
        if len(predecessorList) == 0:
            return set()
        
        # This is the same as getting the identity element of the set of all TACCopy instructions.
        incomingCopies = set(self._getAllCopyInstructions())
        for pred in predecessorList:
            incomingCopies &= pred.reachingCopies

        return incomingCopies
    
    def calculateReachingCopies(self, aliased: set[TACValue]):
        identitySet = set(self._getAllCopyInstructions())
        nodeStack: list[ControlFlowNode] = []

        # We'll be analyzing the nodes in reverse postorder. This makes sure that we'll normally 
        # operate on a block whose input transfer copies have already been calculated; in other 
        # words, the transfer copies of all the predecessors have been calculated. This though will 
        # not always the case in graph with loops, but it's as good as it gets.
        for nodeID in self.traverse():
            if nodeID < 0:
                # Skip the entry and exit blocks.
                continue
            node = self._getNodeByID(nodeID)
            nodeStack.append(node)
            # Initialize the reaching copies of all nodes with the identity set.
            node.reachingCopies = identitySet

        # Operate until there are no more changes in the incoming nodes.
        while len(nodeStack) > 0:
            node = nodeStack.pop()
            # Store the current reaching copies.
            startingReachingCopies = node.reachingCopies
            # Calculate the reaching copies with the current graph information.
            incomingCopies = self._calculateReachingCopiesToNode(node)
            node.anotateReachingCopies(incomingCopies, aliased)
            if startingReachingCopies != node.reachingCopies:
                # The current analysis changed this node incoming copies, therefore, all successors
                # have to be recalculated. Only add them if they're not in the stack.
                for succID in node.successors:
                    if succID < 0: 
                        continue
                    succ = self._getNodeByID(succID)
                    if succ not in nodeStack:
                        nodeStack.insert(0, succ)

    """
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    DEAD STORE ELIMINATION
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    """
    def _calculateLiveVariablesFromNode(self, node: ControlFlowNode) -> set[TACValue]:
        successorList = [self._getNodeByID(id) for id in list(node.successors)]
        
        if len(successorList) == 0:
            return set()
        
        # All variables which are alive at the start of the successors must be alive at the end of 
        # the parent node. 
        liveVars: set[TACValue] = set()
        for succ in successorList:
            liveVars |= succ.liveVariables

        return liveVars
    
    def calculateLiveVariables(self, aliased: set[TACValue]):
        nodeStack: list[ControlFlowNode] = []

        # We'll be analyzing the nodes in postorder. This makes sure that we'll normally operate on 
        # a block with all successors liveness calculated.
        for nodeID in reversed(self.traverse()):
            if nodeID < 0:
                # Skip the entry and exit blocks.
                continue
            node = self._getNodeByID(nodeID)
            nodeStack.append(node)
            # Initialize the reaching copies of all nodes with the null set.
            node.reachingCopies = set()

        # We'll suppose that all aliased variables will be live at the exit node.
        self.exitNode.liveVariables = aliased

        # Operate until there are no more changes in the nodes.
        while len(nodeStack) > 0:
            node = nodeStack.pop()
            # Store the current live variables.
            startingLiveVariables = node.liveVariables
            # Calculate the live variables with the current graph information.
            liveVariables = self._calculateLiveVariablesFromNode(node)
            node.anotateLiveVariables(liveVariables, aliased)
            if startingLiveVariables != node.liveVariables:
                # The current analysis changed this node, therefore, all predecessors have to be 
                # recalculated. Only add them if they're not in the stack.
                for succID in node.predecessors:
                    if succID < 0: 
                        continue
                    succ = self._getNodeByID(succID)
                    if succ not in nodeStack:
                        nodeStack.insert(0, succ)

