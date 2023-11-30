program main
    use jacobi_method, only: jacobi, cujacobi, accjacobi

    implicit none

    logical, parameter :: verbose = .false.
    character(4) :: run_mode = 'cpu'      ! 'cuda' | 'acc', otherwise 'cpu'

    real :: start_time, stop_time

    integer :: icount, size
    integer, parameter :: file_id = 10
    character*100 :: condfile

    double precision, allocatable :: A(:,:), b(:), x(:)

    icount = iargc()

    if ( icount >= 1 ) then
        call getarg(1, condfile)  ! The first argument
        if ( icount >= 2 ) then
            call getarg(2, run_mode)  ! The second argument
        endif
    else
        write(*,*) "Input File not specified."
        stop
    endif

    print *, 'Reading data: ', trim(condfile)
    open(file_id, file=trim(condfile), status='old', action='read')

    read(file_id, *) size
    print *, 'Matrix size: ', size, new_line('')

    allocate(A(size,size), b(size), x(size))

    read(file_id, *) A
    if (verbose) then
        print *, 'A: '
        print *, '    [', A(1,1), ' ... ',A(1,size),' ]'
        print *, '    [ ... ]'
        print *, '    [', A(size,1), ' ... ',A(size,size),' ]'
    endif

    read(file_id, *) b
    if (verbose) then
        print *, 'b: '
        print *, '    [', b(1), ' ... ',b(size),' ]', new_line('')
    endif

    close(file_id)

    x = 0.0d0


    if (trim(run_mode) == 'cuda') then
        call print_header('Jacobi Method (CUDA)')
        call cpu_time(start_time)
        call cujacobi(A, b, x, size, size)
        call cpu_time(stop_time)
    elseif (trim(run_mode) == 'acc') then
        call print_header('Jacobi Method (OpenACC)')
        call cpu_time(start_time)
        call accjacobi(A, b, x, size, size)
        call cpu_time(stop_time)
    else
        call print_header('Jacobi Method (CPU)')
        call cpu_time(start_time)
        call jacobi(A, b, x, size, size)
        call cpu_time(stop_time)
    endif


    print *, 'Time elapsed: ', (stop_time - start_time) * 1000, ' ms', new_line('')

    if (verbose) then
        print *, 'Solution (x): '
        print *, '    [', x(1), ' ... ',x(size),' ]', new_line('')
    endif

    deallocate(A, b, x)

end program main

subroutine print_header(title)
    implicit none
    character(len=*), intent(in) :: title
    print *, '========================================'
    print *, title
    print *, '========================================'
end subroutine print_header
