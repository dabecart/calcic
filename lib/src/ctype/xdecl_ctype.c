/***************************************************************************************************
 * xdecl_ctype.c
 * 
 * Contains all declarations of global variables used by the ctype functions.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_CTYPE
#include <ctype.h>

const unsigned char _CTYPES_TABLE[128] = {
  _BB,  _BB,  _BB,  _BB,  _BB,  _BB,  _BB,  _BB,
  _BB,  _CN,  _CN,  _CN,  _CN,  _CN,  _BB,  _BB,
  _BB,  _BB,  _BB,  _BB,  _BB,  _BB,  _BB,  _BB,
  _BB,  _BB,  _BB,  _BB,  _BB,  _BB,  _BB,  _BB,
  _SP,  _PU,  _PU,  _PU,  _PU,  _PU,  _PU,  _PU,
  _PU,  _PU,  _PU,  _PU,  _PU,  _PU,  _PU,  _PU,
  NUM,  NUM,  NUM,  NUM,  NUM,  NUM,  NUM,  NUM,
  NUM,  NUM,  _PU,  _PU,  _PU,  _PU,  _PU,  _PU,
  _PU,  AUP,  AUP,  AUP,  AUP,  AUP,  AUP,  _UP,
  _UP,  _UP,  _UP,  _UP,  _UP,  _UP,  _UP,  _UP,
  _UP,  _UP,  _UP,  _UP,  _UP,  _UP,  _UP,  _UP,
  _UP,  _UP,  _UP,  _PU,  _PU,  _PU,  _PU,  _PU,
  _PU,  ALO,  ALO,  ALO,  ALO,  ALO,  ALO,  _LO,
  _LO,  _LO,  _LO,  _LO,  _LO,  _LO,  _LO,  _LO,
  _LO,  _LO,  _LO,  _LO,  _LO,  _LO,  _LO,  _LO,
  _LO,  _LO,  _LO,  _PU,  _PU,  _PU,  _PU,  _BB,
};