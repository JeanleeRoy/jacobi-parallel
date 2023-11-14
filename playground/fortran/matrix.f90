program matrix

    implicit none

    integer :: n, m
    real, allocatable :: mat(:,:)

    write (*,*) "Enter the number of rows and columns"
    print *, "rows: "
    read *, n
    print *, "columns: "
    read *, m

    allocate(mat(n,m))

    ! fill with random values from 1 to 10
    call random_number(mat)
    mat = 10*mat + 1

    ! print the matrix
    print '(A,/)', "The matrix is:"
    call print_matrix(n, m, mat)

    deallocate(mat)

end program matrix

subroutine print_matrix(n, m, mat)

    implicit none

    integer, intent(in) :: n, m
    real, intent(in) :: mat(n,m)

    integer :: i

    do i = 1, n
        print *, mat(i,:)
    end do
    print *

end subroutine print_matrix
