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
#define CS_INT                  'd'
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

static long ulongToString(unsigned long val, int writeSign, int base, int upper, char *str, size_t maxLen) {
    if(maxLen < 2) {
        return -1;
    }

    long digitsWritten = 0;

    // Write the number from right to left, starting at str[maxLen - 1].
    char *buf = str + (maxLen - 1);
    do{
        uldiv_t div = uldiv(val, base);
        *buf = numberToChar(div.rem, upper);
        val = div.quot;
        
        buf--;
    }while((str <= buf) && (val > 0));

    if(val != 0) {
        // There wasn't enough space to represent 'val'.
        return -1;
    }

    // Write the + sign if required. Check that there's enough space for it.
    if(writeSign) {
        if(buf < str) {
            return -2;
        }

        *buf = '+';
        digitsWritten++;
        buf--;
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

static long longToString(long val, int writeSign, int base, int upper, char *str, size_t maxLen) {
    if(maxLen < 2) {
        return -1;
    }

    long digitsWritten = 0;
    if(val < 0) {
        *str = '-';
        str++;
        maxLen--;
        digitsWritten++;

        // No need to print the sign inside the 'ulongToString' call.
        writeSign = 0;
        // Get the number without the sign.
        val = -val;
    }

    digitsWritten += ulongToString(val, writeSign, base, upper, str, maxLen);

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

static long doubleToFixedPointString(double val, int writeSign, int upper, int precision, 
    char *str, size_t maxLen) 
{
    if(maxLen < 4) {
        return -1;
    }

    long digitsWritten = 0;

    // Print the sign.
    *str = 0;
    // Use the 'signbit' macro to detect -0.0.
    if(signbit(val)) {
        *str = '-';
        // Get the number without the sign.
        val = -val;
    }else if(writeSign) {
        *str = '+';
    }
    
    if(*str != 0) {
        // We wrote a sign symbol. Advance the pointers.
        str++;
        digitsWritten++;
    }

    int specialCases = handleSpecialDoubleValues(val, upper, str);
    if(specialCases != 0) {
        return digitsWritten + specialCases;
    }

    double divider = 1.0;
    // This is so that we print at least a '0.xx' at the beginning. 'decimalPlaces' = 0 is reserved
    // for the period sign. A negative value is for the integer digits, and positive for the 
    // decimals.
    int decimalPlaces = -1;
    if(val >= 10.0) {
        int log_val = ilog10(val);
        divider = pow10(log_val);
        // Subtract 1 for the period sign.
        decimalPlaces = -log_val - 1;
    }

    do{
        if(decimalPlaces == 0) {
            *str = '.';
        }else {
            // Move the digit to print to the units position.
            double div = val / divider;
            int digit;
            if(decimalPlaces == precision) {
                // The last decimal must be rounded.
                digit = round(div);
            }else {
                // The rest of decimal values must be floored.
                digit = div;
            }
            // Remove the digit from the value.
            val -= digit * divider;
            // Advance for the next divider.
            divider /= 10.0;

            *str = numberToChar(MIN(round(digit),9), upper);
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

static long doubleToScientificString(double val, int writeSign, int upper, int precision, 
    char *str, size_t maxLen) 
{
    if(maxLen < 4) {
        return -1;
    }

    long digitsWritten = 0;

    // Print the sign.
    *str = 0;
    // Use the 'signbit' macro to detect -0.0.
    if(signbit(val)) {
        *str = '-';
        // Get the number without the sign.
        val = -val;
    }else if(writeSign) {
        *str = '+';
    }
    
    if(*str != 0) {
        // We wrote a sign symbol. Advance the pointers.
        str++;
        digitsWritten++;
    }

    int specialCases = handleSpecialDoubleValues(val, upper, str);
    if(specialCases != 0) {
        return digitsWritten + specialCases;
    }

    // Zero is another special case.
    if(val == 0.0) {
        *str = '0';
        if(precision > 0) {
            str++;
            *str = '.';
            str++;
            for(int decimals = 0; decimals < precision; decimals++) {
                *str = '0';
                str++;
            }
            *str = upper ? 'E' : 'e';
            str++;
            *str = '0';
            str++;
            *str = '0';
            return 5 + precision;
        }
    }

    int decimalPlaces = -1;
    int log_val = ilog10(val);

    int dividerExponent = log_val;
    double divider = pow10(dividerExponent);

    do{
        if(decimalPlaces == 0) {
            *str = '.';
        }else {
            // Move the digit to print to the units position.
            double div = val / divider;
            int digit;
            if(decimalPlaces == precision) {
                // The last decimal must be rounded.
                digit = round(div);
            }else {
                // The rest of decimal values must be floored.
                digit = div;
            }
            // Remove the digit from the value.
            val -= digit * divider;
            // Next divider.
            dividerExponent--;
            divider = pow10(dividerExponent);

            *str = numberToChar(MIN(digit, 9), upper);
        }
        
        str++;
        digitsWritten++;
        decimalPlaces++;
    }while((digitsWritten < maxLen) && (decimalPlaces <= precision));

    if(decimalPlaces <= precision) {
        // There wasn't enough space to represent all digits.
        return -1;
    }

    *str = upper ? 'E' : 'e';
    str++;
    digitsWritten++;

    // TODO: add leading zeros
    digitsWritten += longToString(log_val, 1, 10, upper, str, maxLen - digitsWritten);

    return digitsWritten;
}


long _generateFormattedString(const char *format, va_list args, char *out, size_t maxLen) {
    // 'out' may be NULL. In this case, do not write anything to it.
    int writeToOut = (out != NULL);
    // Leave space for the null termination.
    maxLen--;

    char *str = out;
    char * const end_str = out + maxLen;

    // Used to store temporary conversions from "value" to string.
    char temp[64];

    long freeSpace = maxLen;
    while(*format != 0) {
        freeSpace = end_str - str;
        // Do not write if we're passed 'end_str'.
        writeToOut &= (freeSpace > 0);

        // Normal characters.
        if(*format != '%') {
            if(writeToOut) {
                *str = *format;
            }

            str++;
            format++;
            continue;
        }

        // Escape sequence.
        format++;

        if(*format == '%') {
            // %% scape sequence -> %.
            if(writeToOut) {
                *str = '%';
            }

            str++;
            format++;
            continue;
        }

        // Get length modifiers.
        int lenMod = LEN_MOD_NONE;
        int toIncFormat = 1;
        char escapeChar = *format;
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

        // Fetch from the variadic arguments and convert to string, save them in 'temp'.
        long writtenChars = 0;
        if(convSpecifier == CS_INT) {
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
            }else {
                // Error, the escape char is not valid for this type. Go back to the start of the 
                // escape sequence and return an error.
                return BAD_FORMAT_STRING;
            }
            writtenChars = longToString(x, 0, 10, 0, temp, sizeof(temp));
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

            writtenChars = ulongToString(x, 0, base, convSpecifier == CS_UINT_HEX_UPPER, temp, sizeof(temp));
        }else if((convSpecifier == CS_DOUBLE_LOWER) || (convSpecifier == CS_DOUBLE_UPPER)) {
            double x = va_arg(args, double);
            writtenChars = doubleToFixedPointString(x, 0, convSpecifier == CS_DOUBLE_UPPER, 6, temp, sizeof(temp));
        }else if((convSpecifier == CS_SCIENT_LOWER) || (convSpecifier == CS_SCIENT_UPPER)) {
            double x = va_arg(args, double);
            writtenChars = doubleToScientificString(x, 0, convSpecifier == CS_SCIENT_UPPER, 6, temp, sizeof(temp));
        }

        // Transfer from 'temp' to the output string.
        long toWrite = MIN(freeSpace, writtenChars);
        if(toWrite > 0) {
            if(writeToOut) {
                memcpy(str, temp, writtenChars);
            }
            str += toWrite;
        }
    }

    if(out != NULL) {
        // Add the null terminator at the end. Do not take it into account when calculating the 
        // return value of the function.
        *str = 0;
    }

    return str - out;
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