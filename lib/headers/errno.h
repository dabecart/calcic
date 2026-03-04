/***************************************************************************************************
 * <errno.h>
 * 
 * Used to report error conditions, according to Sectio 7.5 of the C99 standard.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#ifndef _CALCIC_ERRNO_h
#define _CALCIC_ERRNO_h

#define EDOM    1
#define EILSEQ  2
#define ERANGE  3

extern int errno;

#endif // _CALCIC_ERRNO_h