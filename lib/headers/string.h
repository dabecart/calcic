/***************************************************************************************************
 * <string.h>
 * 
 * Useful macros and functions for manipulating arrays of character type, as defined in section 7.21
 * of the C99 standard.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#ifndef _CALCIC_STRING_h
#define _CALCIC_STRING_h

#include <stddef.h> // size_t, NULL

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Library aliases.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#define memcpy(s1, s2, n)    __builtin_memcpy(s1, s2, n)
#define memmove(s1, s2, n)   __builtin_memmove(s1, s2, n)
#define strcpy(s1, s2)       __builtin_strcpy(s1, s2)
#define strncpy(s1, s2, n)   __builtin_strncpy(s1, s2, n)

#define strcat(s1, s2)       __builtin_strcat(s1, s2)
#define strncat(s1, s2, n)   __builtin_strncat(s1, s2, n)

#define memcmp(s1, s2, n)    __builtin_memcmp(s1, s2, n)
#define strcmp(s1, s2)       __builtin_strcmp(s1, s2)
#define strncmp(s1, s2, n)   __builtin_strncmp(s1, s2, n)

#define memchr(s, c, n)      __builtin_memchr(s, c, n)
#define strchr(s, c)         __builtin_strchr(s, c)
#define strcspn(s1, s2)      __builtin_strcspn(s1, s2)
#define strpbrk(s1, s2)      __builtin_strpbrk(s1, s2)
#define strrchr(s, c)        __builtin_strrchr(s, c)
#define strspn(s1, s2)       __builtin_strspn(s1, s2)
#define strstr(s1, s2)       __builtin_strstr(s1, s2)
#define strtok(s1, s2)       __builtin_strtok(s1, s2)

#define memset(s, c, n)      __builtin_memset(s, c, n)
#define strlen(s)            __builtin_strlen(s)

// TODO: add restrict qualifiers.

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Copying functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

// Copies 'n' bytes from source 's2' to destination 's1'. Memory areas must not overlap.
void *__builtin_memcpy(void *s1, const void *s2, size_t n);

// Copies 'n' bytes from source 's2' to destination 's1'. Safe for overlapping memory.
void *__builtin_memmove(void *s1, const void *s2, size_t n);

// Copies the string 's2' into 's1', including the null terminator.
char *__builtin_strcpy(char* s1, const char *s2);

// Copies up to 'n' characters from string 's2' into 's1'.
char *__builtin_strncpy(char *s1, const char *s2, size_t n);

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Concatenation functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

// Appends string 's2' to the end of string 's1'.
char *__builtin_strcat(char *s1, char *s2);

// Appends up to 'n' characters of string 's2' to the end of string 's1'.
char *__builtin_strncat(char *s1, char *s2, size_t n);

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Comparison functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

// Compares the first 'n' bytes of memory areas 's1' and 's2'.
int __builtin_memcmp(const void *s1, const void *s2, size_t n);

// Compares string 's1' against string 's2'.
int __builtin_strcmp(const char *s1, const char *s2);

// TODO: strcoll not implemented.

// Compares up to the first 'n' characters of strings 's1' and 's2'.
int __builtin_strncmp(const char *s1, const char *s2, size_t n);

// TODO: strxfrm not implemented.

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Search functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

// Searches the first 'n' bytes of memory 's' for the character 'c'.
void *__builtin_memchr(const void *s, int c, size_t n);

// Finds the first occurrence of character 'c' in string 's'.
char *__builtin_strchr(const char *s, int c);

// Returns the length of the initial segment of 's1' that contains NO characters from 's2'.
size_t __builtin_strcspn(const char *s1, const char *s2);

// Finds the first occurrence in string 's1' of any character from string 's2'.
char *__builtin_strpbrk(const char *s1, const char *s2);

// Finds the last occurrence of character 'c' in string 's'.
char *__builtin_strrchr(const char *s, int c);

// Returns the length of the initial segment of 's1' that consists ONLY of characters from 's2'.
size_t __builtin_strspn(const char *s1, const char *s2);

// Finds the first occurrence of substring 's2' within string 's1'.
char *__builtin_strstr(const char *s1, const char *s2);

// Breaks string 's1' into tokens separated by any of the characters in string 's2'.
char *__builtin_strtok(char *s1, const char *s2);

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Miscellaneous functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

// Fills the first 'n' bytes of memory area 's' with the constant byte 'c'.
void *__builtin_memset(void *s, int c, size_t n);

// TODO: strerror not implemented.

// Returns the length of string 's' (excluding the null terminator).
size_t __builtin_strlen(const char *s);

#endif // _CALCIC_STRING_h