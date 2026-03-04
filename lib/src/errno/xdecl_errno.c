/***************************************************************************************************
 * xdecl_errno.c
 * 
 * Contains all declarations of global variables used by the errno library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include "errno.h"

// At startup, errno is 0.
int errno = 0;