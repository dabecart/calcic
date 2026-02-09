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

            cfg = ControlFlowGraph(optInsts)

            if self.flags.unreachable_code_elimination:
                cfg = self.runUnreachableCodeElimination(cfg)
            if self.flags.copy_propagation:
                cfg = self.runCopyPropagation(cfg)
            if self.flags.dead_store_elimination:
                cfg = self.runDeadStoreElimination(cfg)

            optInsts = cfg.toInstructionList()

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
        # >> Delete unreachable nodes. 
        # For that, traverse the graph keeping track of the nodes nodes visited.
        visitedNodes = cfg.traverse()
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

    def runCopyPropagation(self, cfg: ControlFlowGraph) -> ControlFlowGraph:
        return cfg

    def runDeadStoreElimination(self, cfg: ControlFlowGraph) -> ControlFlowGraph:
        return cfg

@dataclass(eq=False)
class ControlFlowNode:
    nodeID: int
    instructions: list[TACInstruction]      = field(default_factory=list)
    predecessors: set[int]                  = field(default_factory=set)
    successors: set[int]                    = field(default_factory=set)

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
    
    # Use to traverse the graph and run a function for each node. Returns a set of all the visited 
    # node's IDs.
    def traverse(self) -> set[int]:
        visitedNodes: set[int] = set()
        nodeStack: list[ControlFlowNode] = [self.entryNode]

        while nodeStack:
            currentNode = nodeStack.pop()

            if currentNode not in visitedNodes:
                visitedNodes.add(currentNode.nodeID)

            # Add successors to stack (reversed to maintain original order). Skip those that were
            # already visited.
            succs = [nodeId 
                 for nodeId in reversed(list(currentNode.successors)) 
                 if nodeId not in visitedNodes]
            for s in succs:
                if s == -1:
                    nodeStack.append(self.entryNode)
                elif s == -2:
                    nodeStack.append(self.exitNode)
                else:
                    nodeStack.append(self.nodes[s])

        return visitedNodes

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
                case TACJumpIfZero() | TACJumpIfValue():
                    # Connect to the target node and the next one, the target will be reached in 
                    # case the condition is met, and the next node otherwise.
                    targetNode = self.nodes[self.names[finalInst.target]]
                    node.connectSuccessor(targetNode)
                    node.connectSuccessor(nextNode)
                case _:
                    node.connectSuccessor(nextNode)
