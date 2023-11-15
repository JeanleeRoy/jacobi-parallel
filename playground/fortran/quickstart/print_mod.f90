! Each module should be written in a separate .f90 source file.
! Modules need to be compiled prior to any program units that use them.

module print_mod
    implicit none

    private  ! All entities are now module-private by default
    public num_value, print_matrix  ! Explicitly declare public entities

    real, parameter :: num_value = 2.0
    ! integer :: private_var

contains

    subroutine print_matrix(A)
        implicit none

        real, dimension(:,:), intent(in) :: A
        integer :: i

        do i = 1, size(A,1) ! size(A,1) is the number of rows of A (first dimension)
            write(*,*) A(i,:)
        end do

    end subroutine print_matrix

end module print_mod
