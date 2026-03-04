/***************************************************************************************************
 * <ctype.h>
 * 
 * Utils for classifying and mapping characters, as written in section 7.4 of the C99 standard.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#ifndef _CALCIC_CTYPE_h
#define _CALCIC_CTYPE_h

int isalnum(int c);
int isalpha(int c);
int isblank(int c);
int iscntrl(int c);
int isdigit(int c);
int isgraph(int c);
int islower(int c);
int isprint(int c);
int ispunct(int c);
int isspace(int c);
int isupper(int c);
int isxdigit(int c);
int tolower(int c);
int toupper(int c);

#ifdef COMPILING_CTYPE

#define _BB		0x80  // BEL, BS, etc.
#define _CN		0x40  // CR, FF, HT, NL, VT
#define _DI		0x20  // '0'-'9'
#define _PU		0x10  // Punctuation
#define _SP		0x08  // Space
#define _LO		0x04  // 'a'-'z'
#define _UP		0x02  // 'A'-'Z'
#define _HX		0x01  // '0'-'9', 'A'-'F', 'a'-'f' (hexadecimal)

#define NUM (_DI|_HX) // Numbers
#define ALO (_LO|_HX) // Lowercase letters
#define AUP (_UP|_HX) // Uppercase letters

extern const unsigned char _CTYPES_TABLE[128];

// Used to get the flags inside the '_CTYPES_TABLE' table.
// Note that if EOF is inputted, it will also return 0.
#define TYPES(x) ((((x) >= 0) || ((x) < 128)) ? _CTYPES_TABLE[(x)] : 0)

#endif

#endif // _CALCIC_CTYPE_h