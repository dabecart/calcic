"""
types_x64.py

Types supported by the x64 architecture. Classifies types according the System V ABI.

calcic. Written by @dabecart, 2026.
"""

from __future__ import annotations

import enum

from src.TAC import *
from src.calcic_types import *
from src.builtin.builtin_functions_TAC import *

class AssemblyBaseType(enum.Enum):
    DOUBLE   = "sd"
    FLOAT    = "ss"
    QUADWORD = "q"
    LONGWORD = "l"
    WORD     = "w"
    BYTE     = "b"

    # Not a real type, but used to represent an unmapped region of memory.
    BYTEARRAY = "?"

class AssemblyClassType(enum.Enum):
    # Written in order of decreasing priority.
    MEMORY   = enum.auto()
    INTEGER  = enum.auto()
    SSE      = enum.auto()   # Float or double.
    NO_CLASS = enum.auto()   # Unclassified/padding.

# Used to classify eightbytes. Bytesize might be irregular.
class AssemblyClass:
    def __init__(self, classType: AssemblyClassType, byteSize: int) -> None:
        self.classType = classType
        self.byteSize = byteSize

    # To enable sorting in the enum classifying algorithm. 
    def __lt__(self, other):
        if not isinstance(other, AssemblyClass):
            raise ValueError()
        return self.classType.value < other.classType.value

    def toAssemblyType(self) -> AssemblyType:
        if self.classType == AssemblyClassType.SSE:
            match self.byteSize:
                case 8: asmbType = AssemblyType.DOUBLE
                case 4: asmbType = AssemblyType.FLOAT
                case _: raise ValueError()

        elif self.classType == AssemblyClassType.INTEGER:
            match self.byteSize:
                case 8: asmbType = AssemblyType.QUADWORD
                case 4: asmbType = AssemblyType.LONGWORD
                case 2: asmbType = AssemblyType.WORD
                case 1: asmbType = AssemblyType.BYTE
                # For non standard types (3, 5, 6 and 7 bytes)...
                case _: asmbType = AssemblyType(AssemblyBaseType.BYTEARRAY, self.byteSize, 0)
        else:
            asmbType = AssemblyType(AssemblyBaseType.BYTEARRAY, self.byteSize, 0)

        return asmbType

class AssemblyType:
    DOUBLE: AssemblyType
    FLOAT: AssemblyType
    QUADWORD: AssemblyType
    LONGWORD: AssemblyType
    WORD: AssemblyType
    BYTE: AssemblyType

    def __init__(self, baseType: AssemblyBaseType, size: int, alignment: int, sectionName: str = "",
                 members: list[AssemblyClass] = []) -> None:
        self.baseType = baseType
        self.size = size
        self.alignment = alignment
        self.sectionName = sectionName
        self.members = members

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, AssemblyType):
            return False
        return (self.baseType == value.baseType) and (self.size == value.size) and (self.alignment == value.alignment)

    def isQuad(self) -> bool:
        return self.baseType in (AssemblyBaseType.DOUBLE, AssemblyBaseType.QUADWORD)
    
    def isDecimal(self) -> bool:
        return self.baseType in (AssemblyBaseType.DOUBLE, AssemblyBaseType.FLOAT)
    
    def isScalar(self) -> bool:
        return self.baseType in (AssemblyBaseType.QUADWORD, AssemblyBaseType.LONGWORD, AssemblyBaseType.WORD, AssemblyBaseType.BYTE)

    def getDataSectionName(self) -> str:
        if self.baseType == AssemblyBaseType.BYTEARRAY:
            raise ValueError()
        return self.sectionName
    
    def getTypeName(self) -> str:
        return self.baseType.value

    @staticmethod
    def _classifyArray(arrayDeclarator: ArrayDeclaratorType) -> list[AssemblyClass]:
        # TODO: Think if alignment is to be considered here.
        arrayBaseType = AssemblyType.fromTAC(arrayDeclarator.getArrayBaseType())
        return arrayBaseType.members * arrayDeclarator.size

    @staticmethod
    def _classifyStruct(structDeclarator: BaseDeclaratorType) -> list[AssemblyClass]:
        # Classify the structure.
        memberClass: list[AssemblyClass] = []
        structSize = structDeclarator.getByteSize()
        if structSize > 16:
            fullEightBytes = structSize // 8
            memberClass.extend([AssemblyClass(AssemblyClassType.MEMORY, 8)] * fullEightBytes)
            finalEightByte = structSize % 8
            if finalEightByte != 0:
                memberClass.append(AssemblyClass(AssemblyClassType.MEMORY, finalEightByte))
        else:
            # Convert the struct into a flat array of fields.
            flattenedMembers = structDeclarator.baseType.flattenStruct()
            # Convert these fields into assembly classes.
            asmbClasses: list[AssemblyClass] = []
            for mem in flattenedMembers:
                if mem == TypeSpecifier.VOID.toBaseType():
                    # This is padding.
                    asmbClasses.append(AssemblyClass(AssemblyClassType.NO_CLASS, 1))
                else:
                    asmbMembers = AssemblyType.fromTAC(mem).members
                    # To make the next step easier, split all the members into single byte members.
                    for subMember in asmbMembers:
                        asmbClasses.extend([AssemblyClass(subMember.classType, 1)] * subMember.byteSize)
            # Now group them into eightbytes.
            byteCount: int = 0
            currentClass: AssemblyClassType = AssemblyClassType.NO_CLASS
            for cl in asmbClasses:
                byteCount += 1

                # Stick to the most strict class type.
                if cl.classType.value < currentClass.value:
                    currentClass = cl.classType

                if byteCount == 8:
                    # New eightbyte.
                    memberClass.append(AssemblyClass(currentClass, 8))
                    byteCount = 0
                    currentClass = AssemblyClassType.NO_CLASS

            # There may be an incomplete eightbyte at the end.
            if byteCount != 0:
                memberClass.append(AssemblyClass(currentClass, byteCount))
        
        return memberClass
    
    @staticmethod
    def _classifyUnion(unionDeclarator: BaseDeclaratorType) -> list[AssemblyClass]:
        memberClass: list[AssemblyClass] = []

        unionSize = unionDeclarator.getByteSize()
        if unionSize > 16:
            fullEightBytes = unionSize // 8
            memberClass.extend([AssemblyClass(AssemblyClassType.MEMORY, 8)] * fullEightBytes)
            finalEightByte = unionSize % 8
            if finalEightByte != 0:
                memberClass.append(AssemblyClass(AssemblyClassType.MEMORY, finalEightByte))
        else:
            # Get the classes of each member but split them into 1 byte classes. This will make the 
            # matching algorithm easier to follow.
            classifiedMembers: list[list[AssemblyClass]] = []
            greatestMemberListCount: int = 0
            for member in unionDeclarator.baseType.getMembers():
                # Get the assembly type of the union member, this will calculate its members.
                asmbType = AssemblyType.fromTAC(member.type)
                # Now, split them into 1 byte classes.
                singleByteMemberList: list[AssemblyClass] = []
                for innerMember in asmbType.members:
                    singleByteMemberList.extend([AssemblyClass(innerMember.classType, 1)] * innerMember.byteSize)
                # Add to the outer loop list.
                classifiedMembers.append(singleByteMemberList)
                # Monitor which list of assembly members is bigger for the next for loop.
                if len(singleByteMemberList) > greatestMemberListCount:
                    greatestMemberListCount = len(singleByteMemberList)
            
            # Compare each byte of the classified members and get the most restrictive.
            byteClasses: list[AssemblyClass] = []
            for byteIndex in range(greatestMemberListCount):
                # Join all the classes at this byteIndex in an array and sort it.
                currentByteClasses: list[AssemblyClass] = []
                for memberIndex in range(len(classifiedMembers)):
                    # Check limits.
                    if byteIndex >= len(classifiedMembers[memberIndex]):
                        continue
                    
                    currentByteClasses.append(classifiedMembers[memberIndex][byteIndex])
                
                currentByteClasses.sort()
                # The first is the highest priority class of this eightbyte.
                byteClasses.append(AssemblyClass(currentByteClasses[0].classType, 1))

            # Finally, group the bytes into eightbytes.
            byteCount: int = 0
            currentClass: AssemblyClassType = AssemblyClassType.NO_CLASS
            for cl in byteClasses:
                byteCount += 1

                # Stick to the most strict class type.
                if cl.classType.value < currentClass.value:
                    currentClass = cl.classType

                if byteCount == 8:
                    # New eightbyte.
                    memberClass.append(AssemblyClass(currentClass, 8))
                    byteCount = 0
                    currentClass = AssemblyClassType.NO_CLASS

            # There may be an incomplete eightbyte at the end.
            if byteCount != 0:
                memberClass.append(AssemblyClass(currentClass, byteCount))
        
        return memberClass

    @staticmethod
    def fromTAC(tac: DeclaratorType) -> AssemblyType:
        if isinstance(tac, BaseDeclaratorType):
            match tac.baseType.name:
                case "CHAR" | "SIGNED_CHAR" | "UCHAR":   
                    return AssemblyType.BYTE
                case "SHORT" | "USHORT":  
                    return AssemblyType.WORD
                case "INT" | "UINT":    
                    return AssemblyType.LONGWORD
                case "LONG" | "ULONG":   
                    return AssemblyType.QUADWORD
                case "DOUBLE": 
                    return AssemblyType.DOUBLE
                case "FLOAT": 
                    return AssemblyType.FLOAT
                case "STRUCT":
                    return AssemblyType(
                            AssemblyBaseType.BYTEARRAY, 
                            tac.getByteSize(), tac.getAlignment(),
                            members=AssemblyType._classifyStruct(tac))
                case "UNION":
                    return AssemblyType(
                            AssemblyBaseType.BYTEARRAY, 
                            tac.getByteSize(), tac.getAlignment(),
                            members=AssemblyType._classifyUnion(tac))
                case "ENUM":
                    return AssemblyType.LONGWORD
                case _:
                    raise ValueError(f"Cannot convert {tac} to AssemblyType")
        elif isinstance(tac, ArrayDeclaratorType):
            byteLen = tac.getByteSize()
            # As the System V x64 ABI says...
            if byteLen > 16:
                # Alignment is always 16 when the size of the array is greater than 16.
                alignment = 16
            else:
                # Alignment is the same as the array base type.
                alignment = tac.getArrayBaseType().getByteSize()

            return AssemblyType(
                AssemblyBaseType.BYTEARRAY, 
                byteLen, alignment, 
                members=AssemblyType._classifyArray(tac))
        else:
            # Both pointers and functions are quad values in 64 bit systems.
            return AssemblyType.QUADWORD

# Common assembly types.
AssemblyType.DOUBLE      = AssemblyType(AssemblyBaseType.DOUBLE,    8, 8, "double", [AssemblyClass(AssemblyClassType.SSE,     8)])
AssemblyType.FLOAT       = AssemblyType(AssemblyBaseType.FLOAT,     4, 4, "float",  [AssemblyClass(AssemblyClassType.SSE,     4)])
AssemblyType.QUADWORD    = AssemblyType(AssemblyBaseType.QUADWORD,  8, 8, "quad",   [AssemblyClass(AssemblyClassType.INTEGER, 8)])
AssemblyType.LONGWORD    = AssemblyType(AssemblyBaseType.LONGWORD,  4, 4, "long",   [AssemblyClass(AssemblyClassType.INTEGER, 4)])
AssemblyType.WORD        = AssemblyType(AssemblyBaseType.WORD,      2, 2, "word",   [AssemblyClass(AssemblyClassType.INTEGER, 2)])
AssemblyType.BYTE        = AssemblyType(AssemblyBaseType.BYTE,      1, 1, "byte",   [AssemblyClass(AssemblyClassType.INTEGER, 1)])
