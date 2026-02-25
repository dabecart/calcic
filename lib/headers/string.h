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

// TODO: add restrict qualifier.

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Copying functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

// Copies 'n' bytes from source 's2' to destination 's1'. Memory areas must not overlap.
void *memcpy(void *s1, const void *s2, size_t n);

// Copies 'n' bytes from source 's2' to destination 's1'. Safe for overlapping memory.
void *memmove(void *s1, const void *s2, size_t n);

// Copies the string 's2' into 's1', including the null terminator.
char *strcpy(char* s1, const char *s2);

// Copies up to 'n' characters from string 's2' into 's1'.
char *strncpy(char *s1, const char *s2, size_t n);

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Concatenation functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

// Appends string 's2' to the end of string 's1'.
char *strcat(char *s1, char *s2);

// Appends up to 'n' characters of string 's2' to the end of string 's1'.
char *strncat(char *s1, char *s2, size_t n);

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Comparison functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

// Compares the first 'n' bytes of memory areas 's1' and 's2'.
int memcmp(const void *s1, const void *s2, size_t n);

// Compares string 's1' against string 's2'.
int strcmp(const char *s1, const char *s2);

// strcoll not implemented.

// Compares up to the first 'n' characters of strings 's1' and 's2'.
int strncmp(const char *s1, const char *s2, size_t n);

// strxfrm not implemented.

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Search functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

// Searches the first 'n' bytes of memory 's' for the character 'c'.
void *memchr(const void *s, int c, size_t n);

// Finds the first occurrence of character 'c' in string 's'.
char *strchr(const char *s, int c);

// Returns the length of the initial segment of 's1' that contains NO characters from 's2'.
size_t strcspn(const char *s1, const char *s2);

// Finds the first occurrence in string 's1' of any character from string 's2'.
char *strpbrk(const char *s1, const char *s2);

// Finds the last occurrence of character 'c' in string 's'.
char *strrchr(const char *s, int c);

// Returns the length of the initial segment of 's1' that consists ONLY of characters from 's2'.
size_t strspn(const char *s1, const char *s2);

// Finds the first occurrence of substring 's2' within string 's1'.
char *strstr(const char *s1, const char *s2);

// Breaks string 's1' into tokens separated by any of the characters in string 's2'.
char *strtok(char *s1, const char *s2);

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Miscellaneous functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

// Fills the first 'n' bytes of memory area 's' with the constant byte 'c'.
void *memset(void *s, int c, size_t n);

// strerror not implemented.

// Returns the length of string 's' (excluding the null terminator).
size_t strlen(const char *s);

#endif // _CALCIC_STRING_h