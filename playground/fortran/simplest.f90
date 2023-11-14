program main

   implicit none

   character*50 :: greeting = 'Hello, World!'
   character*8 :: bye = 'Goodbye!'
   character*7 :: name = 'Jeanlee'
   integer :: i

   do i = 0, 10, 2   ! Iterate over even numbers from 0 to 10
      if (i.lt.10) then
         print *, trim(greeting), i
      else
         print *, bye//' '//name, i
      endif
   enddo

end program
