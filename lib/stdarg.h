#ifndef calcic_stdarg_h
#define calcic_stdarg_h

#define va_start(a,b) __builtin_va_start(a,b)
#define va_arg(a,b) __builtin_va_arg(a,b)
#define va_end(a) __builtin_va_end(a)

typedef __builtin_va_list va_list;

#endif // calcic_stdarg_h