# 0 "src/ctype/isspace.c"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "src/ctype/isspace.c"
# 10 "src/ctype/isspace.c"
# 1 "/home/dano/repos/calcic/lib/headers/ctype.h" 1 3 4
# 12 "/home/dano/repos/calcic/lib/headers/ctype.h" 3 4

# 12 "/home/dano/repos/calcic/lib/headers/ctype.h" 3 4
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
# 42 "/home/dano/repos/calcic/lib/headers/ctype.h" 3 4
extern const unsigned char _CTYPES_TABLE[128];
# 11 "src/ctype/isspace.c" 2


# 12 "src/ctype/isspace.c"
int isspace(int c) {
    return 
# 13 "src/ctype/isspace.c" 3 4
          ((((
# 13 "src/ctype/isspace.c"
          c
# 13 "src/ctype/isspace.c" 3 4
          ) >= 0) || ((
# 13 "src/ctype/isspace.c"
          c
# 13 "src/ctype/isspace.c" 3 4
          ) < 128)) ? _CTYPES_TABLE[(
# 13 "src/ctype/isspace.c"
          c
# 13 "src/ctype/isspace.c" 3 4
          )] : 0) 
# 13 "src/ctype/isspace.c"
                   & (
# 13 "src/ctype/isspace.c" 3 4
                      0x40 
# 13 "src/ctype/isspace.c"
                          | 
# 13 "src/ctype/isspace.c" 3 4
                            0x08
# 13 "src/ctype/isspace.c"
                               );
}
