// O_*, F_*, FD_* bit values for Linux.
// Taken from: /usr/include/x86_64-linux-gnu/bits/fcntl-linux.h

#ifndef FCNTL_h
#define FCNTL_h

#define O_ACCMODE     0003
#define O_RDONLY      00
#define O_WRONLY      01
#define O_RDWR        02
#define O_CREAT       0100	
#define O_EXCL        0200	
#define O_NOCTTY      0400	
#define O_TRUNC       01000	
#define O_APPEND      02000
#define O_NONBLOCK    04000
#define O_NDELAY      
#define O_SYNC        04010000
#define O_FSYNC       O_SYNC
#define O_ASYNC       020000
#define __O_LARGEFILE 

#define __O_DIRECTORY   0200000
#define __O_NOFOLLOW    0400000
#define __O_CLOEXEC     02000000
#define __O_DIRECT      040000
#define __O_NOATIME     01000000
#define __O_PATH        010000000
#define __O_DSYNC       010000
#define __O_TMPFILE     (020000000 | __O_DIRECTORY)

#endif // FCNTL_h