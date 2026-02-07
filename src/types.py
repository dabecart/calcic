"""
types.py

Different C types and declarators used widely in the compiler. 

calcic. Written by @dabecart, 2026.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import enum
from dataclasses import dataclass

# Util to store parameter info, such as those of functions, structs or unions.
@dataclass
class ParameterInformation:
    type: DeclaratorType
    name: str

    # Used for the parameters of structs.
    offset: int = 0

    def __str__(self) -> str:
        return f"{self.type} {self.name}"
    
    def __eq__(self, other):
        if not isinstance(other, ParameterInformation):
            return False
        return self.type == other.type

# The storage class defines the lifetime, scope, and visibility of variables.
class StorageClass(enum.Enum):
    STATIC = "static"
    EXTERN = "extern"
    # TODO: auto, register

# Type qualifiers advise the compiler about how the variable will be used.
@dataclass
class TypeQualifier:
    const: bool = False
    # TODO: volatile, restrict

    def toSet(self) -> set[str]:
        ret = set()
        if self.const: ret.add("const")
        return ret

    @staticmethod
    def fromSet(inputSet: set[str]) -> TypeQualifier:
        return TypeQualifier("const" in inputSet)

    def contains(self, other: TypeQualifier) -> bool:
        return other.toSet() <= self.toSet()
    
    def union(self, other: TypeQualifier) -> TypeQualifier:
        un = self.toSet() | other.toSet()
        return TypeQualifier.fromSet(un)

    def __str__(self) -> str:
        ret = ""
        if self.const: ret += "const "
        return ret

class TypeSpecifier:
    VOID: TypeSpecifier
    CHAR: TypeSpecifier
    SIGNED_CHAR: TypeSpecifier
    UCHAR: TypeSpecifier
    SHORT: TypeSpecifier
    INT: TypeSpecifier
    LONG: TypeSpecifier
    USHORT: TypeSpecifier
    UINT: TypeSpecifier
    ULONG: TypeSpecifier
    DOUBLE: TypeSpecifier
    FLOAT: TypeSpecifier

    @staticmethod
    def STRUCT(originalIdentifier: str, mangledIdentifier: str, byteSize: int, alignment: int, 
               members: list[ParameterInformation]) -> TypeSpecifier:
        kwargs = dict(identifier = mangledIdentifier, members = members)
        return TypeSpecifier("STRUCT", f"struct {originalIdentifier}", 
                             byteSize, alignment, **kwargs)
    
    @staticmethod
    def UNION(originalIdentifier: str, mangledIdentifier: str, byteSize: int, alignment: int, 
               members: list[ParameterInformation]) -> TypeSpecifier:
        kwargs = dict(identifier = mangledIdentifier, members = members)
        return TypeSpecifier("UNION", f"union {originalIdentifier}", 
                             byteSize, alignment, **kwargs)

    @staticmethod
    def ENUM(originalIdentifier: str) -> TypeSpecifier:
        # Enums are represented by an int (its size and alignment are 4).
        return TypeSpecifier("ENUM", f"enum {originalIdentifier}", 4, 4)

    def __init__(self, name: str, value: str, byteSize: int, alignment: int, **kwargs) -> None:
        self.name: str = name
        self.value: str = value
        self.byteSize: int = byteSize
        self.alignment: int = alignment
        
        # Struct/union stuff.
        self.identifier: str = kwargs.get("identifier", "") # Mangled identifier.
        self.members: list[ParameterInformation] = kwargs.get("members", [])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypeSpecifier):
            return False
        ret = self.name == other.name and self.value == other.value
        if ret and self.name in ("STRUCT", "UNION"):
            # Check the struct and union's declaration, in particular, the mangled identifier.
            ret = self.identifier == other.identifier
        return ret
    
    def __str__(self) -> str:
        return self.value
    
    def toBaseType(self, qualifiers = TypeQualifier()) -> BaseDeclaratorType:
        return BaseDeclaratorType(self, qualifiers)

    def getByteSize(self) -> int:
        return self.byteSize
    
    def getAlignment(self) -> int:
        return self.alignment
            
    def getMembers(self) -> list[ParameterInformation]:
        if self.name not in ("STRUCT", "UNION"):
            raise ValueError("Cannot get member of a non struct/union type")

        return self.members
    
    def getMember(self, memberName: str) -> ParameterInformation|None:
        for member in self.getMembers():
            if member.name == memberName:
                return member
        return None

    def isSignedInt(self) -> bool:
        match self:
            case TypeSpecifier.CHAR | TypeSpecifier.SIGNED_CHAR | \
                 TypeSpecifier.SHORT | TypeSpecifier.INT | TypeSpecifier.LONG:      
                return True
            case _:
                return False
            
    def isInteger(self) -> bool:
        return self in (TypeSpecifier.CHAR, TypeSpecifier.UCHAR, TypeSpecifier.SIGNED_CHAR,
                        TypeSpecifier.SHORT, TypeSpecifier.USHORT, 
                        TypeSpecifier.INT, TypeSpecifier.UINT, 
                        TypeSpecifier.LONG, TypeSpecifier.ULONG) or \
               self.name == "ENUM"

    def isDecimal(self) -> bool:
        return self in (TypeSpecifier.DOUBLE, TypeSpecifier.FLOAT)
    
    def isCharacter(self) -> bool:
        return self in (TypeSpecifier.CHAR, TypeSpecifier.UCHAR, TypeSpecifier.SIGNED_CHAR) 
    
    def needsIntegerPromotion(self) -> bool:
        return self in (TypeSpecifier.CHAR, TypeSpecifier.UCHAR, TypeSpecifier.SIGNED_CHAR, 
                        TypeSpecifier.SHORT, TypeSpecifier.USHORT) or \
               self.name == "ENUM"

    def flattenStruct(self) -> list[DeclaratorType]:
        if self.name != "STRUCT":
            raise ValueError(f"Cannot flatten {self.name}")
        
        ret: list[DeclaratorType] = []
        
        # Padding will be defined using void.
        def addPadding(byteCount: int):
            ret.extend([TypeSpecifier.VOID.toBaseType()] * byteCount)
        
        lastOffset: int = 0
        member: ParameterInformation
        for member in self.members:
            addPadding(member.offset - lastOffset)
            lastOffset = member.offset + member.type.getByteSize()

            if isinstance(member.type, BaseDeclaratorType) and member.type.baseType.name == "STRUCT":
                ret.extend(member.type.baseType.flattenStruct())
            elif isinstance(member.type, ArrayDeclaratorType):
                arrayBaseType = member.type.getArrayBaseType()
                totalArrayLength = member.type.getByteSize() // arrayBaseType.getByteSize()
                ret.extend([arrayBaseType] * totalArrayLength)
            else:
                ret.append(member.type)

        # Add the final padding.
        addPadding(self.getByteSize() - lastOffset)

        return ret

TypeSpecifier.VOID          = TypeSpecifier("VOID",         "void",         -1, -1)
TypeSpecifier.CHAR          = TypeSpecifier("CHAR",         "char",          1,  1)
TypeSpecifier.SIGNED_CHAR   = TypeSpecifier("SIGNED_CHAR",  "signed char",   1,  1)
TypeSpecifier.UCHAR         = TypeSpecifier("UCHAR",        "unsigned char", 1,  1)
TypeSpecifier.SHORT         = TypeSpecifier("SHORT",        "short",         2,  2)
TypeSpecifier.USHORT        = TypeSpecifier("USHORT",       "unsigned short",2,  2)
TypeSpecifier.INT           = TypeSpecifier("INT",          "int",           4,  4)
TypeSpecifier.UINT          = TypeSpecifier("UINT",         "unsigned int",  4,  4)
TypeSpecifier.LONG          = TypeSpecifier("LONG",         "long",          8,  8)
TypeSpecifier.ULONG         = TypeSpecifier("ULONG",        "unsigned long", 8,  8)
TypeSpecifier.DOUBLE        = TypeSpecifier("DOUBLE",       "double",        8,  8)
TypeSpecifier.FLOAT         = TypeSpecifier("FLOAT",        "float",         4,  4)

# When parsing the "type" of variable being declared, a DeclaratorType will be used. 
# There are three types of declarators:
# - Base types, which are normal types (int, long, float...).
# - Pointer.
# - Function.
# DeclaratorType stores types in a reverse order to normal C.
# An example, int (*param)[3][4] is read as "param is a POINTER to an ARRAY of 3 of an ARRAY of 4 int"
# DeclaratorType is Array(Array(Pointer(Base(int)), 3), 4)
class DeclaratorType(ABC):
    def __init__(self) -> None:
        super().__init__()

    # Returns a new decayed declarator type (if the innermost type is an array, it is converted to
    # a pointer).
    @abstractmethod
    def decay(self) -> DeclaratorType:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __str__(self) -> str:
        return super().__str__()
    
    @abstractmethod
    def unqualify(self) -> BaseDeclaratorType:
        pass

    @abstractmethod
    def getTypeQualifiers(self) -> TypeQualifier:
        pass

    @abstractmethod
    def setTypeQualifiers(self, newQualifiers: TypeQualifier):
        pass

    @abstractmethod
    def copy(self) -> PointerDeclaratorType:
        pass

    def __repr__(self) -> str:
        return self.__str__()

    def isInteger(self) -> bool:
        if isinstance(self, BaseDeclaratorType):
            return self.baseType.isInteger()
        return False

    def isDecimal(self) -> bool:
        if isinstance(self, BaseDeclaratorType):
            return self.baseType.isDecimal()
        return False

    def getByteSize(self) -> int:
        if isinstance(self, BaseDeclaratorType):
            return self.baseType.getByteSize()
        elif isinstance(self, PointerDeclaratorType):
            # TODO: For 64 bit systems...
            return 8
        elif isinstance(self, ArrayDeclaratorType):
            return self.declarator.getByteSize() * self.size

        raise ValueError("Not implemented")
    
    def getAlignment(self) -> int:
        if isinstance(self, BaseDeclaratorType):
            return self.baseType.getAlignment()
        elif isinstance(self, PointerDeclaratorType):
            # TODO: For 64 bit systems...
            return 8
        elif isinstance(self, ArrayDeclaratorType):
            # ABI says that for arrays bigger or equal to 16 bytes, the alignment is 16. 
            # return 16 if self.getByteSize() >= 16 else self.declarator.getAlignment()
            # TODO: is the rule above necessary for structs?
            return self.declarator.getAlignment()

        raise ValueError("Not implemented")

    def isSigned(self) -> bool:
        if isinstance(self, BaseDeclaratorType):
            return self.baseType.isSignedInt()
        return False
    
    # Traverses an array type 
    def getArrayBaseType(self) -> DeclaratorType:
        if isinstance(self, ArrayDeclaratorType):
            return self.declarator.getArrayBaseType()
        else:
            return self
    
    def isComplete(self) -> bool:
        # Void is always incomplete.
        if isinstance(self, BaseDeclaratorType) and self.baseType == TypeSpecifier.VOID:
            return False
        # If the byte size cannot be calculated or is negative, then it's an incomplete type.
        try:
            byteSize = self.getByteSize()
        except:
            return False
        return byteSize >= 0
    
    def isScalar(self) -> bool:
        if isinstance(self, (ArrayDeclaratorType, FunctionDeclaratorType)) or \
           self == TypeSpecifier.VOID.toBaseType() or \
           (isinstance(self, BaseDeclaratorType) and self.baseType.name in ("STRUCT", "UNION")):
            return False
        return True
    
    def isArithmetic(self) -> bool:
        if isinstance(self, BaseDeclaratorType):
            return self.baseType.name not in ("STRUCT", "UNION", "VOID")
        return False
    
class BaseDeclaratorType(DeclaratorType):
    def __init__(self, baseType: TypeSpecifier, qualifiers: TypeQualifier = TypeQualifier()) -> None:
        self.baseType = baseType
        self.qualifiers = qualifiers
        super().__init__()

    def decay(self) -> DeclaratorType:
        # Base types do not decay.
        return BaseDeclaratorType(self.baseType, self.qualifiers)

    def __str__(self) -> str:
            return f"{self.qualifiers}{self.baseType}"
    
    def __eq__(self, other):
        if not isinstance(other, BaseDeclaratorType):
            return False
        return self.baseType == other.baseType and self.qualifiers == other.qualifiers
    
    def copy(self) -> BaseDeclaratorType:
        return BaseDeclaratorType(self.baseType, self.qualifiers)
    
    def unqualify(self) -> BaseDeclaratorType:
        ret = self.copy()
        ret.qualifiers = TypeQualifier()
        return ret

    def getTypeQualifiers(self) -> TypeQualifier:
        return self.qualifiers

    def setTypeQualifiers(self, newQualifiers: TypeQualifier):
        self.qualifiers = newQualifiers

class PointerDeclaratorType(DeclaratorType):
    def __init__(self, declarator: DeclaratorType, qualifiers: TypeQualifier = TypeQualifier()) -> None:
        self.declarator = declarator
        self.qualifiers = qualifiers
        super().__init__()

    def decay(self) -> DeclaratorType:
        # Pointers do not decay.
        return PointerDeclaratorType(self.declarator, self.qualifiers)
    
    def copy(self) -> PointerDeclaratorType:
        return PointerDeclaratorType(self.declarator, self.qualifiers)
        
    def __str__(self) -> str:
            return f"{self.qualifiers}pointer to {self.declarator}"
    
    def __eq__(self, other):
        if not isinstance(other, PointerDeclaratorType):
            return False
        return self.declarator == other.declarator and self.qualifiers == other.qualifiers
    
    def unqualify(self) -> PointerDeclaratorType:
        ret = self.copy()
        ret.qualifiers = TypeQualifier()
        return ret
    
    def getTypeQualifiers(self) -> TypeQualifier:
        return self.qualifiers
    
    def setTypeQualifiers(self, newQualifiers: TypeQualifier):
        self.qualifiers = newQualifiers

class ArrayDeclaratorType(DeclaratorType):
    def __init__(self, declarator: DeclaratorType, size: int) -> None:
        self.declarator = declarator
        self.size = size
        super().__init__()

    def decay(self) -> DeclaratorType:
        # An array in C decays to pointer to their first element.
        # For example, int[3][4] decays to int (*)[4].
        # Another example, *int[4] decays to int**.
        return PointerDeclaratorType(self.declarator, self.getTypeQualifiers())
    
    def copy(self) -> ArrayDeclaratorType:
        return ArrayDeclaratorType(self.declarator, self.size)

    def __str__(self) -> str:
        return f"array[{self.size}] of {self.declarator}"
    
    def __eq__(self, other):
        if not isinstance(other, ArrayDeclaratorType):
            return False
        return self.declarator == other.declarator and self.size == other.size
    
    def unqualify(self) -> ArrayDeclaratorType:
        return ArrayDeclaratorType(self.declarator.unqualify(), self.size)
    
    def getTypeQualifiers(self) -> TypeQualifier:
        return self.declarator.getTypeQualifiers()
    
    def setTypeQualifiers(self, newQualifiers: TypeQualifier):
        self.qualifiers = newQualifiers

class FunctionDeclaratorType(DeclaratorType):
    def __init__(self, params: list[ParameterInformation], declarator: DeclaratorType) -> None:
        self.params = params
        self.returnDeclarator = declarator
        super().__init__()

    def decay(self) -> DeclaratorType:
        raise ValueError()

    def copy(self) -> FunctionDeclaratorType:
        return FunctionDeclaratorType(self.params, self.returnDeclarator)

    def __str__(self) -> str:
        paramStrings = [str(p) for p in self.params]
        return f"{self.returnDeclarator}({', '.join(paramStrings)})"
    
    def __eq__(self, other):
        if not isinstance(other, FunctionDeclaratorType):
            return False
        return self.returnDeclarator == other.returnDeclarator and all([p1 == p2 for p1, p2 in zip(self.params, other.params)])
    
    def unqualify(self) -> FunctionDeclaratorType:
        raise ValueError()
    
    def getTypeQualifiers(self) -> TypeQualifier:
        raise ValueError()

    def setTypeQualifiers(self, newQualifiers: TypeQualifier):
        raise ValueError()
    
@dataclass
class DeclaratorInformation:
    name: str
    type: DeclaratorType
    params: list[ParameterInformation]