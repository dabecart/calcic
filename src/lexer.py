import re

keywordPatterns = [
    ("char",                re.compile(r"char\b")),
    ("short",               re.compile(r"short\b")),
    ("int",                 re.compile(r"int\b")),
    ("long",                re.compile(r"long\b")),
    ("signed",              re.compile(r"signed\b")),
    ("unsigned",            re.compile(r"unsigned\b")),
    ("double",              re.compile(r"double\b")),
    ("float",               re.compile(r"float\b")),
    ("return",              re.compile(r"return\b")),
    ("void",                re.compile(r"void\b")),
    ("if",                  re.compile(r"if\b")),
    ("else",                re.compile(r"else\b")),
    ("do",                  re.compile(r"do\b")),
    ("while",               re.compile(r"while\b")),
    ("for",                 re.compile(r"for\b")),
    ("break",               re.compile(r"break\b")),
    ("continue",            re.compile(r"continue\b")),
    ("switch",              re.compile(r"switch\b")),
    ("case",                re.compile(r"case\b")),
    ("default",             re.compile(r"default\b")),
    ("goto",                re.compile(r"goto\b")),
    ("static",              re.compile(r"static\b")),
    ("extern",              re.compile(r"extern\b")),
    ("sizeof",              re.compile(r"sizeof\b")),
    ("struct",              re.compile(r"struct\b")),
    ("union",               re.compile(r"union\b")),
    ("const",               re.compile(r"const\b")),
]

tokenPatterns = [
    ("identifier",              re.compile(r"[a-zA-Z_]\w*\b")),
    ("constant",                re.compile(r"(0[xX][0-9A-Fa-f]+|0[bB][01]+|0[0-7]+|[0-9]+)(?=[^\w.])")),
    ("long_constant",           re.compile(r"((0[xX][0-9A-Fa-f]+|0[bB][01]+|0[0-7]+|[0-9]+)[lL])(?=[^\w.])")),
    ("unsigned_constant",       re.compile(r"((0[xX][0-9A-Fa-f]+|0[bB][01]+|0[0-7]+|[0-9]+)[uU])(?=[^\w.])")),
    ("unsigned_long_constant",  re.compile(r"((0[xX][0-9A-Fa-f]+|0[bB][01]+|0[0-7]+|[0-9]+)(?:[uU][lL]|[lL][uU]))(?=[^\w.])")),
    ("double_constant",         re.compile(r"(([0-9]*\.[0-9]+|[0-9]+\.?)[Ee][+-]?[0-9]+|[0-9]*\.[0-9]+|[0-9]+\.)(?=[^\w.])")),
    ("float_constant",          re.compile(r"((([0-9]*\.[0-9]+|[0-9]+\.?)[Ee][+-]?[0-9]+|[0-9]*\.[0-9]+|[0-9]+\.)[fF])(?=[^\w.])")),
    ("double_constant", re.compile(r"("
            # Hexadecimal Double: 0x, hex digits/point, mandatory p/P exponent
            r"0[xX]([0-9a-fA-F]*\.[0-9a-fA-F]+|[0-9a-fA-F]+\.?)[pP][+-]?[0-9]+|"
            # Decimal Double: digits/point, optional e/E exponent OR mandatory point
            r"([0-9]*\.[0-9]+|[0-9]+\.?)[Ee][+-]?[0-9]+|[0-9]*\.[0-9]+|[0-9]+\."
    r")(?=[^\w.])")),

    ("float_constant", re.compile(r"("
            # Hexadecimal Float: 0x, hex digits/point, mandatory p/P exponent, [fF] suffix
            r"0[xX]([0-9a-fA-F]*\.[0-9a-fA-F]+|[0-9a-fA-F]+\.?)[pP][+-]?[0-9]+[fF]|"
            # Decimal Float: digits/point, optional e/E exponent, [fF] suffix
            r"(([0-9]*\.[0-9]+|[0-9]+\.?)[Ee][+-]?[0-9]+|[0-9]*\.[0-9]+|[0-9]+\.)[fF]"
    r")(?=[^\w.])")),
    ("character",               re.compile(r"'([^\\\n']|\\.)*'")),
    ("string",                  re.compile(r'"([^\\\n"]|\\.)*"')),
    ("(",                       re.compile(r"\(")),
    (")",                       re.compile(r"\)")),
    ("{",                       re.compile(r"{")),
    ("}",                       re.compile(r"}")),
    ("[",                       re.compile(r"\[")),
    ("]",                       re.compile(r"\]")),
    (";",                       re.compile(r";")),
    ("~",                       re.compile(r"~")),
    ("?",                       re.compile(r"\?")),
    (":",                       re.compile(r":")),
    (",",                       re.compile(r",")),

    (".",                       re.compile(r"\.(?!\d)")), # The next character should NOT be a digit.
    ("->",                      re.compile(r"->")),

    ("+",                       re.compile(r"\+")),
    ("-",                       re.compile(r"-")),
    ("*",                       re.compile(r"\*")),
    ("/",                       re.compile(r"/")),
    ("%",                       re.compile(r"%")),
    ("<<",                      re.compile(r"<<")),
    (">>",                      re.compile(r">>")),
    ("&",                       re.compile(r"&")),
    ("^",                       re.compile(r"\^")),
    ("|",                       re.compile(r"\|")),

    ("=",                       re.compile(r"=")),
    ("+=",                      re.compile(r"\+=")),
    ("-=",                      re.compile(r"-=")),
    ("*=",                      re.compile(r"\*=")),
    ("/=",                      re.compile(r"/=")),
    ("%=",                      re.compile(r"%=")),
    ("<<=",                     re.compile(r"<<=")),
    (">>=",                     re.compile(r">>=")),
    ("&=",                      re.compile(r"&=")),
    ("^=",                      re.compile(r"\^=")),
    ("|=",                      re.compile(r"\|=")),

    ("++",                      re.compile(r"\+\+")),
    ("--",                      re.compile(r"--")),

    ("!",                       re.compile(r"!")),
    ("&&",                      re.compile(r"&&")),
    ("||",                      re.compile(r"\|\|")),
    ("==",                      re.compile(r"==")),
    ("!=",                      re.compile(r"!=")),
    ("<",                       re.compile(r"<")),
    (">",                       re.compile(r">")),
    ("<=",                      re.compile(r"<=")),
    (">=",                      re.compile(r">=")),
]

lineMarkings = re.compile(
    r'^#\s+(\d+)\s+"([^"]+)"(?:\s+(.*))?$'
)

class Token:
    TYPE_SPECIFIER: list[str] = ["struct", "union", "void", "char", "short", "int", "long", 
                                 "unsigned", "signed", "double", "float"]
    CAST_SPECIFIER: list[str] = ["const"] + TYPE_SPECIFIER
    SPECIFIER: list[str] = ["static", "extern", "const"] + TYPE_SPECIFIER
    
    def __init__(self, id: str, value: str, file: str = "", line: int = 0, col: int = 0) -> None:
        self.id = id
        self.value = value
        self.file = file
        self.line = line
        self.col = col

    def __str__(self) -> str:
        if self.id == self.value:
            return f"{self.id}"

        return f"{self.id}({self.value})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def getPosition(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"
    
    def parseIntegerToken(self) -> int:
            if self.id == "character":
                # Returns the ASCII value.
                return ord(self.value)

            core = self.value.lower().replace("u", "").replace("l", "")

            # Determine the base
            if core.startswith("0x"):
                # Hexadecimal.
                try:
                    return int(core[2:], 16)
                except:
                    raise ValueError(f"Constant {self.value} cannot be interpreted as a base-16 integer")
            elif core.startswith("0b"):
                # Binary.
                try:
                    return int(core[2:], 2)
                except:
                    raise ValueError(f"Constant {self.value} cannot be interpreted as a base-2 integer")
            elif core.startswith("0") and len(core) > 1:
                # Octal literal (C-style: leading zero).
                try:
                    return int(core, 8)
                except:
                    raise ValueError(f"Constant {self.value} cannot be interpreted as a base-8 integer")
            else:
                # Decimal.
                try:
                    return int(core, 10)
                except:
                    raise ValueError(f"Constant {self.value} cannot be interpreted as a base-10 integer")

    def parseDecimalToken(self) -> float:
        core = self.value.lower().rstrip("f")
        if core.startswith("0x"):
            # Hex decimal.
            return float.fromhex(core)
        # Standard decimal.
        return float(core)

def _decodeCString(raw: str) -> str:
    # Remove the '' or "".
    raw = raw[1:-1]

    out: list[str] = []
    i: int = 0
    n: int = len(raw)

    escapeCharacters = {
        "'": "'",
        '"': '"',
        "?": "?",
        "\\": "\\",
        "a": "\a",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "v": "\v",
    }

    while i < n:
        c = raw[i]

        # Normal character
        if c != "\\":
            out.append(c)
            i += 1
            continue

        # Escape sequence
        i += 1
        if i >= n:
            raise SyntaxError("incomplete escape sequence")

        esc = raw[i]

        # Simple escapes
        if esc in escapeCharacters:
            out.append(escapeCharacters[esc])
            i += 1
            continue

        # Hex escape: \xhh...
        if esc == "x":
            i += 1
            start = i

            while i < n and raw[i] in "0123456789abcdefABCDEF":
                i += 1

            if start == i:
                raise SyntaxError("invalid hex escape sequence")

            value = int(raw[start:i], 16)
            out.append(chr(value & 0xFF))
            continue

        # Octal escape: \[0-7]{1,3}
        if esc in "01234567":
            start = i
            i += 1

            for _ in range(2):
                if i < n and raw[i] in "01234567":
                    i += 1
                else:
                    break

            value = int(raw[start:i], 8)
            out.append(chr(value & 0xFF))
            continue

        # Unknown escape
        raise SyntaxError(f"unknown escape sequence \\{esc}")

    return "".join(out)

def lex(filename: str) -> list[Token]:
    tokens: list[Token] = []

    lineNumber: int = 1
    colNumber: int = 1
    preprocessorFileName: str = filename
    with open(filename, 'r') as file:
        data: str
        while (data := file.readline()) != "":
            # Check if this is a line marking (generated from the preprocessor).
            lineMarkingsMatch = lineMarkings.match(data)
            if lineMarkingsMatch:
                # The line marking tells the number of the NEXT line.
                lineNumber = int(lineMarkingsMatch.group(1)) - 1
                # It also tells the name of the file.
                preprocessorFileName = lineMarkingsMatch.group(2)
                continue

            lineNumber += 1
            colNumber = 1
            while data != "":
                originalLen = len(data)
                data = data.lstrip(" \n\r\t")
                colNumber += originalLen - len(data)

                if data == "":
                    break

                bestMatch = None
                bestMatchID = None
                bestMatchLen = -1

                for id, pat in tokenPatterns:
                    m = pat.match(data)
                    if not m: continue
                    
                    length = len(m.group(0))
                    if length > bestMatchLen:
                        bestMatchLen = length
                        bestMatch = m.group(0)
                        bestMatchID = id

                if bestMatch is None or bestMatchID is None:
                    raise ValueError(f"{preprocessorFileName}:{lineNumber}: Unexpected token {data}")
                
                # Keywords are identifiers.
                if bestMatchID == "identifier":
                    for id, pat in keywordPatterns:
                        m = pat.match(data)
                        if m:
                            # This identifier is a keyword.
                            bestMatch = m.group(0)
                            bestMatchID = id
                            break
                elif bestMatchID == "character":
                    bestMatch = _decodeCString(bestMatch)
                    if len(bestMatch) != 1:
                        raise ValueError(f"{preprocessorFileName}:{lineNumber}: Empty char")
                elif bestMatchID == "string":
                    bestMatch = _decodeCString(bestMatch)

                tok = Token(bestMatchID, bestMatch, preprocessorFileName, lineNumber, colNumber)
                tokens.append(tok)
                data = data[bestMatchLen:]
                colNumber += bestMatchLen

    return tokens