/***************************************************************************************************
 * xdecl_stdio.c
 * 
 * Contains all declarations of global variables used by the stdio functions.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <calcilib.h>
#include <math.h>
#include <ctype.h>

static FILE in  = {0, READ_MODE};
static FILE out = {1, WRITE_MODE|LINE_BUFFERED_MODE};
static FILE err = {2, WRITE_MODE|UNBUFFERED_MODE};

FILE *stdin = &in;
FILE *stdout = &out;
FILE *stderr = &err;

FILE *files[FOPEN_MAX] = {&in, &out, &err};
// We start with stdin, stdout and stderr.
size_t openFiles = 3;

int _processModeString(const char *mode) {
    int flags = 0;
    if(strcmp(mode, "r") == 0)         flags = READ_MODE;
    else if(strcmp(mode, "w") == 0)    flags = WRITE_MODE | TRUNCATE_MODE;
    else if(strcmp(mode, "a") == 0)    flags = WRITE_MODE | APPEND_MODE;
    else if(strcmp(mode, "rb") == 0)   flags = READ_MODE  | BINARY_MODE;
    else if(strcmp(mode, "wb") == 0)   flags = WRITE_MODE | TRUNCATE_MODE  | BINARY_MODE;
    else if(strcmp(mode, "ab") == 0)   flags = WRITE_MODE | APPEND_MODE    | BINARY_MODE;
    else if(strcmp(mode, "r+") == 0)   flags = READ_MODE  | WRITE_MODE;
    else if(strcmp(mode, "w+") == 0)   flags = READ_MODE  | WRITE_MODE     | TRUNCATE_MODE;
    else if(strcmp(mode, "a+") == 0)   flags = READ_MODE  | WRITE_MODE     | APPEND_MODE;
    else if(strcmp(mode, "r+b") == 0)  flags = READ_MODE  | WRITE_MODE     | BINARY_MODE;
    else if(strcmp(mode, "w+b") == 0)  flags = READ_MODE  | WRITE_MODE     | TRUNCATE_MODE | BINARY_MODE;
    else if(strcmp(mode, "a+b") == 0)  flags = READ_MODE  | WRITE_MODE     | APPEND_MODE   | BINARY_MODE;
    return flags;
}

int initBuffer(FILE *f) {
    f->buffer = malloc(BUFSIZ);
    if(f->buffer == NULL) {
        return 0;
    }

    f->bufSize = BUFSIZ;
    f->head = f->tail = f->buffer;
    return 1;
}

int closeBuffer(FILE *f) {
    if(f->buffer != NULL) {
        free(f->buffer);
        f->bufSize = 0;
        f->head = f->tail = NULL;
    }
    return 1;
}

size_t push(FILE *f, const unsigned char item) {
    if(f->len >= f->bufSize) {
        return 0;
    }

    *f->head = item;
    f->head++;
    if(f->head >= (f->buffer + f->bufSize)) {
        f->head = f->buffer;
    } 
    f->len++; 
    return 1;
}

size_t push_back(FILE *f, const unsigned char item) {
    if(f->len >= f->bufSize) {
        return 0;
    }

    f->tail--;
    if(f->tail < f->buffer) {
        f->tail = f->buffer + f->bufSize - 1;
    } 
    *f->tail = item;
    f->len++; 
    return 1;
}

size_t push_N(FILE *f, const unsigned char *items, size_t count) {
    size_t toPush = count;
    while(toPush > 0) {
        if(!push(f, *items)){
            break;
        }

        items++;
        toPush--;
    }
    return count - toPush;
}

size_t pop(FILE *f, unsigned char* outItem) {
    if(f->len < 1) {
        return 0;
    }

    *outItem = *f->tail;
    f->tail++;
    if(f->tail >= (f->buffer + f->bufSize)) {
        f->tail = f->buffer;
    } 
    f->len--;
    return 1;
}

size_t pop_N(FILE *f, unsigned char* outItems, size_t count) {
    size_t toPop = count;
    while(toPop > 0) {
        if(!pop(f, outItems)){
            break;
        }

        outItems++;
        toPop--;
    }
    return count - toPop;
}

// Flags.
#define FLAG_LEFT_JUSTIFIED 0x01    // '-'
#define FLAG_SIGN           0x02    // '+'
#define FLAG_SPACE          0x04    // ' '
#define FLAG_ALT_FORM       0x08    // '#'
#define FLAG_ZERO_PADDING   0x10    // '0'

#define FLAG_UPPERCASE      0x20    // Not represented by a flag symbol, but it's an utility.

// Length modifiers.
#define LEN_MOD_NONE        0
#define LEN_MOD_CHAR        1   // hh
#define LEN_MOD_SHORT       2   // h
#define LEN_MOD_LONG        3   // l
#define LEN_MOD_LONG_LONG   4   // ll
#define LEN_MOD_INTMAX      5   // j
#define LEN_MOD_SIZET       6   // z
#define LEN_MOD_PTRDIFF     7   // t
#define LEN_MOD_LONG_DOUBLE 8   // L

// Conversion specifiers.
#define CS_INT_D                'd'
#define CS_INT_I                'i'
#define CS_UINT                 'u'
#define CS_UINT_OCTAL           'o'
#define CS_UINT_HEX_LOWER       'x'
#define CS_UINT_HEX_UPPER       'X'
#define CS_DOUBLE_LOWER         'f'
#define CS_DOUBLE_UPPER         'F'
#define CS_SCIENT_LOWER         'e'
#define CS_SCIENT_UPPER         'E'
#define CS_DOUBLE_AUTO_LOWER    'g'
#define CS_DOUBLE_AUTO_UPPER    'G'
#define CS_DOUBLE_HEX_LOWER     'a'
#define CS_DOUBLE_HEX_UPPER     'A'
#define CS_CHAR                 'c'
#define CS_STRING               's'
#define CS_POINTER              'p'
#define CS_PRINTED_COUNTER      'n'

#define BAD_FORMAT_STRING       (-1)

// Values used during string formatting.
typedef struct {
    int precision;
    int minFieldWidth;
    int flags;
} FormatOptions;

static char numberToChar(int num, int upper) {
    if(num < 10) {
        return num + '0';
    }
    
    if(upper) {
        return num - 10 + 'A';  
    }else {
        return num - 10 + 'a';
    }
}

static long ulongToString(unsigned long val, int base, FormatOptions options, char *str, size_t maxLen) {
    if(maxLen < 2) {
        return -1;
    }

    long digitsWritten = 0;
    int uppercase = (options.flags & FLAG_UPPERCASE) != 0;
    
    // Write the + sign or [space] if required. Check that there's enough space for it.
    int printSign = (options.flags & (FLAG_SIGN|FLAG_SPACE)) != 0;
    if(printSign) {
        if(options.flags & FLAG_SIGN) {
            *str = '+';
        }else {
            *str = ' ';
        }
        str++;
        digitsWritten++;
    }
    
    int precision = options.precision;
    if(options.flags & FLAG_ZERO_PADDING) {
        // Substitute the zero padding flag with a precision value.
        if(printSign) {
            // If the sign was added, subtract it.
            precision = MAX(precision, options.minFieldWidth - 1);
        }else {
            precision = MAX(precision, options.minFieldWidth);
        }
    }

    if((precision == 0) && (val == 0) && 
       !((options.flags & FLAG_ALT_FORM) != 0 && (base == 8))) {
        // A zero value with zero precision returns nothing.
        // The special case of the alt form for base 8 must return at least a zero.
        return 0;
    }

    // Zero values are not affected by the alternative form flag.
    if((options.flags & FLAG_ALT_FORM) && (val != 0)) {
        if(base == 16) {
            if(uppercase) {
                strcpy(str, "0X");
            }else {
                strcpy(str, "0x");
            }
            str += 2;
            digitsWritten += 2;
            if(options.flags & FLAG_ZERO_PADDING) {
                // 0x counts towards the field width, but should not modify the precision number, 
                // unless we're zero padding. In this case, we trick the program by substituting
                // the padding with precision.
                precision -= 2;
            }
        }else if(base == 8) {
            // Add a zero beforehand. 
            *str = '0';
            str++;
            digitsWritten++;
            // Decrement precision as there's already a zero at the start.
            precision--;
        }
    }

    // Write the number from right to left, starting at str[maxLen - 1].
    char *buf = str + (maxLen - 1);
    do{
        uldiv_t div = uldiv(val, base);
        *buf = numberToChar(div.rem, uppercase);
        val = div.quot;
        
        buf--;
        precision--;
    }while((str <= buf) && (val > 0));

    if(val != 0) {
        // There wasn't enough space to represent 'val'.
        return -1;
    }

    // If there's some precision digits left, add '0'.
    while(precision > 0) {
        *buf = '0';
        buf--;
        precision--;
    }

    // 'buf' now points to the last written character.
    buf++;

    // Shift the array to the left so that the most significant digit is at str[0].
    const ptrdiff_t emptyLeft = buf - str;
    digitsWritten += maxLen - emptyLeft;
    for(ptrdiff_t index = 0; index < digitsWritten; index++) {
        str[index] = str[index + emptyLeft];
    }

    return digitsWritten;
}

static long longToString(long val, int base, FormatOptions options, char *str, size_t maxLen) {
    if(maxLen < 2) {
        return -1;
    }

    long digitsWritten = 0;
    if(val < 0) {
        *str = '-';
        str++;
        maxLen--;
        digitsWritten = 1;

        // No need to print the sign or space inside the 'ulongToString' call.
        options.flags &= ~(FLAG_SIGN | FLAG_SPACE);
        // Get the number without the sign.
        val = -val;

        if(options.flags & FLAG_ZERO_PADDING) {
            // Substitute the zero padding flag with a precision value. Subtract one for the sign.
            options.precision = MAX(options.precision, options.minFieldWidth - 1);
            // Clear the zero padding flag.
            options.flags &= ~FLAG_ZERO_PADDING;
        }
    }

    digitsWritten += ulongToString(val, base, options, str, maxLen);

    return digitsWritten;
}

static int handleSpecialDoubleValues(double val, int upper, char *str) {
    if(isinf(val)) {
        if(upper) {
            strcpy(str, "INF");
        }else {
            strcpy(str, "inf");
        }
        return 3;
    }else if(isnan(val)) {
        if(upper) {
            strcpy(str, "NAN");
        }else {
            strcpy(str, "nan");
        }
        return 3;
    }
    return 0;
}

static long doubleToFixedPointString(double val, FormatOptions options,
    char *str, size_t maxLen) 
{
    // When precision is not specified, it is 6.
    int precision = options.precision;
    if(precision < 0) {
        precision = 6;
    }

    // At least, sign, one unit, period and precision decimals.
    if(maxLen < (3 + precision)) {
        return -1;
    }

    long digitsWritten = 0;
    int uppercase = (options.flags & FLAG_UPPERCASE) != 0;

    // Print the sign.
    *str = 0;
    // Use the 'signbit' macro to detect -0.0.
    if(signbit(val)) {
        *str = '-';
        // Get the number without the sign.
        val = -val;
    }else if(options.flags & FLAG_SIGN) {
        *str = '+';
    }else if(options.flags & FLAG_SPACE) {
        *str = ' ';
    }
    
    if(*str != 0) {
        // We wrote a sign symbol. Advance the pointers.
        str++;
        digitsWritten++;
    }

    int specialCases = handleSpecialDoubleValues(val, uppercase, str);
    if(specialCases != 0) {
        return digitsWritten + specialCases;
    }

    // Scale 'val' to an 18 digit integer (in base 10). The first digit of this number will be at 
    // 10^17 (which is the largest integer representable in an unsigned long).
    unsigned long scaled_val = 0;
    int decimalPlaces = -1;
    unsigned long divider = 100000000000000000UL; // 10^17
    if(val != 0.0) {
        int log_val = ilog10(val);
        double scale = pow10(17 - log_val);
        scaled_val = round(val * scale);
        decimalPlaces = -log_val - 1;
    }

    do{
        if(decimalPlaces == 0) {
            // Alt form forces the decimal separator even with zero precision.
            if(((precision > 0) || ((options.flags & FLAG_ALT_FORM) != 0))) {
                *str = '.';
            }else {
                decimalPlaces++;
                continue;
            }
        }else if(divider > 0) {
            *str = numberToChar(scaled_val / divider, uppercase);
            scaled_val %= divider;
            divider /= 10ULL;
        }else {
            *str = '0';
        }
        
        str++;
        digitsWritten++;
        decimalPlaces++;
    }while((digitsWritten < maxLen) && (decimalPlaces <= precision));

    if(decimalPlaces <= precision) {
        // There wasn't enough space to represent all digits.
        return -1;
    }

    return digitsWritten;
}

static long doubleToScientificString(double val, FormatOptions options,
    char *str, size_t maxLen) 
{
    // When precision is not specified, it is 6.
    int precision = options.precision;
    if(precision < 0) {
        precision = 6;
    }

    // At least, sign, one unit, period, precision decimals, exponent letter ('e' or 'E') and three 
    // exponents characters (sign and two digits).
    if(maxLen < (7 + precision)) {
        return -1;
    }

    long digitsWritten = 0;
    int uppercase = (options.flags & FLAG_UPPERCASE) != 0;

    // Print the sign.
    *str = 0;
    // Use the 'signbit' macro to detect -0.0.
    if(signbit(val)) {
        *str = '-';
        // Get the number without the sign.
        val = -val;
    }else if(options.flags & FLAG_SIGN) {
        *str = '+';
    }else if(options.flags & FLAG_SPACE) {
        *str = ' ';
    }
    
    if(*str != 0) {
        // We wrote a sign symbol. Advance the pointers.
        str++;
        digitsWritten++;
    }

    int specialCases = handleSpecialDoubleValues(val, FLAG_UPPERCASE, str);
    if(specialCases != 0) {
        return digitsWritten + specialCases;
    }

    // Zero is another special case.
    if(val == 0.0) {
        long n = doubleToFixedPointString(val, options, str, maxLen - digitsWritten);
        strcpy(str + n, uppercase ? "E00" : "e00");
        return digitsWritten + n + 3;
    }

    // Scale 'val' to an 18 digit integer (in base 10). The first digit of this number will be at 
    // 10^17 (which is the largest integer representable in an unsigned long).
    int log_val = ilog10(val);
    double scale = pow10(17 - log_val);
    unsigned long scaled_val = round(val * scale);

    // Print the first digit.
    unsigned long divider = 100000000000000000UL; // 10^17
    
    *str = numberToChar(scaled_val / divider, uppercase);
    str++;
    digitsWritten++;

    scaled_val %= divider;
    divider /= 10UL;

    // Alt form forces the decimal separator even with zero precision.
    if((precision > 0) || ((options.flags & FLAG_ALT_FORM) != 0)) {
        *str++ = '.';
        digitsWritten++;
    }

    // Extract decimal digits.
    for(int i = 0; i < precision; i++) {
        // If we run out of precision in the scaled_val, just print '0'.
        if (divider > 0) {
            *str = numberToChar(scaled_val / divider, uppercase);
            scaled_val %= divider;
            divider /= 10UL;
        } else {
            *str = '0';
        }
        str++;
        digitsWritten++;

        if(digitsWritten == maxLen) {
            return -1;
        }
    }

    // Print the exponent.
    *str = uppercase ? 'E' : 'e';
    str++;
    digitsWritten++;
    FormatOptions expOpts = {0};
    expOpts.flags = FLAG_SIGN;
    expOpts.precision = 2;
    long expOk = longToString(log_val, 10, expOpts, str, maxLen - digitsWritten);
    if(expOk < 0) {
        return -1;
    }
    digitsWritten += expOk;
    
    return digitsWritten;
}


long _generateFormattedString(const char *format, va_list args, char *out, size_t maxLen) {
    // 'out' may be NULL. In this case, do not write anything to it.
    int writeToOut = (out != NULL);
    // Leave space for the null termination.
    maxLen--;

    char *strOut = out;
    char * const end_str = out + maxLen;

    // Used to store temporary conversions from "value" to string.
    char TEMP_BUF[64];
    // Stores the theoretical number of bytes written to the output.
    long retCount = 0;

    while(*format != 0) {
        // Do not write to the output in case we exceeded maxLen.
        writeToOut &= (retCount < maxLen);

        // Normal characters.
        if(*format != '%') {
            if(writeToOut) {
                *strOut = *format;
                strOut++;
            }

            format++;
            retCount++;
            continue;
        }

        // This starts an escape sequence.
        format++;
        
        if(*format == '%') {
            // %% -> %.
            if(writeToOut) {
                *strOut = '%';
                strOut++;
            }
            
            format++;
            retCount++;
            continue;
        }

        // An unspecified precision will be negative.
        FormatOptions options = {-1, 0, 0};
        // Normally, we'll use the 'TEMP_BUF' array to store the the value that will be copied into 
        // 'out', but that won't happen for strings (%s).
        char *temp = TEMP_BUF;

        // Flags.
        int toIncFormat = 1;
        char escapeChar;
        while(toIncFormat > 0) {
            escapeChar = *format;
            toIncFormat = 1;

            if(escapeChar == '-') {
                options.flags |= FLAG_LEFT_JUSTIFIED;
                // Clear the zero padding flag.
                options.flags &= ~FLAG_ZERO_PADDING;
            }else if(escapeChar == '+') {
                options.flags |= FLAG_SIGN;
                // Clear the space flag.
                options.flags &= ~FLAG_SPACE;
            }else if(escapeChar == ' ') {
                // Only add the space flag if the sign flag is not present.
                if(!(options.flags & FLAG_SIGN)) {
                    options.flags |= FLAG_SPACE;
                }
            }else if(escapeChar == '#') {
                options.flags |= FLAG_ALT_FORM;
            }else if(escapeChar == '0') {
                // Do not add the zero paddding when the field is left justified.
                if(!(options.flags & FLAG_LEFT_JUSTIFIED)) {
                    options.flags |= FLAG_ZERO_PADDING;
                }
            }else {
                toIncFormat = 0;
            }
            format += toIncFormat;
        }

        // Minimum field width.
        escapeChar = *format;
        if(escapeChar == '*') {
            options.minFieldWidth = va_arg(args, int);
            format++;
            
            if(options.minFieldWidth < 0) {
                // A negative field width is taken as a '-' flag plus the field width.
                options.minFieldWidth = -options.minFieldWidth;
                options.flags |= FLAG_LEFT_JUSTIFIED;
                // Clear the zero padding flag.
                options.flags &= ~FLAG_ZERO_PADDING;
            }
        }else if(isdigit(escapeChar)) {
            char *endptr;
            options.minFieldWidth = strtol(format, &endptr, 10);
            format = endptr;
        }

        // Precision.
        if(*format == '.') {
            format++;
            // A single period means precision zero.
            options.precision = 0;

            escapeChar = *format;
            if(escapeChar == '*') {
                options.precision = va_arg(args, int);
                format++;
            }else if(isdigit(escapeChar)) {
                char *endptr;
                options.precision = strtol(format, &endptr, 10);
                if(options.precision < 0) {
                    // Cannot have negative precision.
                    return -1;
                }
                format = endptr;
            }
        }

        // Get length modifiers.
        int lenMod = LEN_MOD_NONE;
        escapeChar = *format;
        toIncFormat = 1;
        if(escapeChar == 'h') {
            if(format[1] == 'h') {
                lenMod = LEN_MOD_CHAR;
                toIncFormat = 2;
            }else {
                lenMod = LEN_MOD_SHORT;
            }
        }else if(escapeChar == 'l') {
            if(format[1] == 'l') {
                lenMod = LEN_MOD_LONG_LONG;
                toIncFormat = 2;
            }else {
                lenMod = LEN_MOD_LONG;
            }
        }else if(escapeChar == 'j') {
            lenMod = LEN_MOD_INTMAX;
        }else if(escapeChar == 'z') {
            lenMod = LEN_MOD_SIZET;
        }else if(escapeChar == 't') {
            lenMod = LEN_MOD_PTRDIFF;
        }else if(escapeChar == 'L') {
            lenMod = LEN_MOD_LONG_DOUBLE;
        }else{
            // No length modifier.
            toIncFormat = 0;
        }

        // Advance the format after parsing the length modifier.
        format += toIncFormat;

        // Conversion specifiers.
        int convSpecifier = *format;
        format++;

        if((convSpecifier == CS_UINT_HEX_UPPER) || 
           (convSpecifier == CS_DOUBLE_UPPER) || 
           (convSpecifier == CS_SCIENT_UPPER) || 
           (convSpecifier == CS_DOUBLE_AUTO_UPPER) || 
           (convSpecifier == CS_DOUBLE_HEX_UPPER)) 
        {
            options.flags |= FLAG_UPPERCASE;
        }

        // Fetch from the variadic arguments and convert to string, save them in 'temp'.
        long writtenChars = 0;
        if((convSpecifier == CS_INT_D) || (convSpecifier == CS_INT_I)) {
            // Signed integer types.
            long x;
            if(lenMod == LEN_MOD_NONE) {
                // int.
                x = va_arg(args, int);
            } else if(lenMod == LEN_MOD_CHAR) {
                // char.
                x = (char) va_arg(args, int);
            } else if(lenMod == LEN_MOD_SHORT) {
                // short.
                x = (short) va_arg(args, int);
            } else if(lenMod == LEN_MOD_LONG) {
                // long.
                x = va_arg(args, long);
            } else if(lenMod == LEN_MOD_SIZET) {
                // size_t
                x = va_arg(args, size_t);
            } else if(lenMod == LEN_MOD_PTRDIFF) {
                // ptrdiff_t
                x = va_arg(args, ptrdiff_t);
            }else {
                // Error, the escape char is not valid for this type. Go back to the start of the 
                // escape sequence and return an error.
                return BAD_FORMAT_STRING;
            }

            writtenChars = longToString(x, 10, options, temp, sizeof(TEMP_BUF));
        }else if((convSpecifier == CS_UINT) || (convSpecifier == CS_UINT_OCTAL) || 
                 (convSpecifier == CS_UINT_HEX_LOWER) || (convSpecifier == CS_UINT_HEX_UPPER)) 
        {
            // Unsigned integer types.
            unsigned long x;
            if(lenMod == LEN_MOD_NONE) {
                // unsigned int.
                x = va_arg(args, unsigned int);
            } else if(lenMod == LEN_MOD_CHAR) {
                // unsigned char.
                x = (unsigned char) va_arg(args, unsigned int);
            } else if(lenMod == LEN_MOD_SHORT) {
                // unsigned short.
                x = (unsigned short) va_arg(args, unsigned int);
            } else if(lenMod == LEN_MOD_LONG) {
                // unsigned long.
                x = va_arg(args, unsigned long);
            } else if(lenMod == LEN_MOD_SIZET) {
                // size_t
                x = va_arg(args, size_t);
            } else if(lenMod == LEN_MOD_PTRDIFF) {
                // ptrdiff_t
                x = va_arg(args, ptrdiff_t);
            }else {
                // Error, the escape char is not valid for this type. Go back to the start of the 
                // escape sequence and return an error.
                return BAD_FORMAT_STRING;
            }

            int base = 10;
            if(convSpecifier == CS_UINT_OCTAL) {
                base = 8;
            }else if((convSpecifier == CS_UINT_HEX_LOWER) || (convSpecifier == CS_UINT_HEX_UPPER)) {
                base = 16;
            }

            writtenChars = ulongToString(x, base, options, temp, sizeof(TEMP_BUF));

        }else if((convSpecifier == CS_DOUBLE_LOWER) || (convSpecifier == CS_DOUBLE_UPPER)) {
            double x = va_arg(args, double);
            writtenChars = doubleToFixedPointString(x, options, temp, sizeof(TEMP_BUF));
        
        }else if((convSpecifier == CS_SCIENT_LOWER) || (convSpecifier == CS_SCIENT_UPPER)) {
            double x = va_arg(args, double);
            writtenChars = doubleToScientificString(x, options, temp, sizeof(TEMP_BUF));
        
        }else if((convSpecifier == CS_DOUBLE_AUTO_LOWER) || (convSpecifier == CS_DOUBLE_AUTO_UPPER)) {
            // Scientific form is used only if the exponent is less than -4 or greater or equal to 
            // the precission.
            double x = va_arg(args, double);
            // Calculate the exponent.
            int log_val = ilog10(x);
            if((log_val < -4) || (log_val >= options.precision)) {
                writtenChars = doubleToScientificString(x, options, temp, sizeof(TEMP_BUF));
            }else {
                writtenChars = doubleToFixedPointString(x, options, temp, sizeof(TEMP_BUF));
            }

        }else if(convSpecifier == CS_CHAR) {
            // Use the pointer as temporary buffer. This will be copied to the 'out' string.
            char c = va_arg(args, int);
            *temp = c;
            writtenChars = 1;

        }else if(convSpecifier == CS_STRING) {
            // Use the pointer as temporary buffer. This will be copied to the 'out' string.
            temp = va_arg(args, char*);
            // Precision dictates the maximum number of bytes to be written.
            writtenChars = strlen(temp);
            if(options.precision > 0) {
                writtenChars = MIN(writtenChars, options.precision);
            }
        
        }else if(convSpecifier == CS_POINTER) {
            // Pointers get written as uppercase hexadecimal with preceding 0x.
            void *ptr = va_arg(args, void*);
            writtenChars = ulongToString((unsigned long) ptr, 16, options, temp, sizeof(TEMP_BUF));
        
        }else if(convSpecifier == CS_PRINTED_COUNTER) {
            // Store in a pointer argument the current number of characters printed.
            int *ptr = va_arg(args, int*);
            *ptr = retCount;
            continue;

        }else {
            return BAD_FORMAT_STRING;
        }
        
        // Get the real width of the field, taking into account the minimum field width.
        long fieldWidth = MAX(writtenChars, options.minFieldWidth);
        // The number of bytes to write depend on the free space.
        long toWrite = MIN(fieldWidth, maxLen - retCount);

        if((toWrite > 0) && writeToOut) {
            const char padSymbol = (options.flags & FLAG_ZERO_PADDING) ? '0' : ' ';
            if(options.flags & FLAG_LEFT_JUSTIFIED) {
                // Copy from the temporary buffer.
                long toCopy = MIN(writtenChars, maxLen - retCount);
                memcpy(strOut, temp, toCopy);
                strOut += toCopy;
                
                // Add the necessary padding.
                for(; toCopy < toWrite; toCopy++) {
                    *strOut = padSymbol;
                    strOut++;
                }
            }else {
                // Add the necessary padding.
                long toPad = MAX(options.minFieldWidth - writtenChars, 0);
                for(long count = 0; count < toPad; count++) {
                    *strOut = padSymbol;
                    strOut++;
                }

                // Copy from the temporary buffer.
                long toCopy = toWrite - toPad;
                memcpy(strOut, temp, toCopy);
                strOut += toCopy;
                
            }
        }

        retCount += fieldWidth;
    }

    if(out != NULL) {
        // Add the null terminator at the end. Do not take it into account when calculating the 
        // return value of the function.
        *strOut = 0;
    }

    return retCount;
}

long _printfToStream(FILE *stream, const char * format, va_list args) {
    char *s = malloc(FORMATTED_STRING_LEN_GUESS);
    if(s == NULL) {
        return -158479;
    }

    long strLen = _generateFormattedString(format, args, s, FORMATTED_STRING_LEN_GUESS);

    if(strLen > FORMATTED_STRING_LEN_GUESS) {
        // The initial guess was wrong... You'll have to create the string again in a bigger buffer.
        // Use the value 'strLen' as the right size. 
        free(s);
        s = malloc(strLen + 1); // Add one as the function does not count the null termination.
        if(s == NULL) {
            return -1;
        }
        strLen = _generateFormattedString(format, args, s, FORMATTED_STRING_LEN_GUESS);
    }

    if(strLen > 0) {
        // Write s into the stream. Return the number of bytes written to the stream.
        strLen = fwrite(s, strLen + 1, 1, stream);
    }

    free(s);

    return strLen;
}