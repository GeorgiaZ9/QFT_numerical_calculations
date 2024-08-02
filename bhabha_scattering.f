
Program bhabha_scattering

    implicit none
    
    real(8), dimension(10) :: p1, p2, p3, p4
    real(8) :: amplitude, s, t, u
    real(8) :: e
    integer :: i
    
    e = 1.0D0  
    
    Print *, "Pass parameters for Feynmann diagram p1, p2, p3, p4: "
    read *, (p1(i), i=1, 4)
    read *, p2(1), p2(2), p2(3), p2(4)
    read *, p3(1), p3(2), p3(3), p3(4)
    read *, p4(1), p4(2), p4(3), p4(4)
    
    do i=1,4
        Print *, p1(i)
    end do 

    do i=1,4
        Print *, p2(i)
    end do 

    do i=1,4
        Print *, p3(i)
    end do

    do i=1,4
        Print *, p4(i)
    end do 
    
    call mandlestam(p1, p2, p3, p4, s, t, u)
    
    amplitude = 2.0D0 * e * ( (s/t)**2 + (t/s)**2 + u**2*(1/s + 1/t)**2)
    
    Print *, "The scattering amplitude is calculated: "
    
    Print *, amplitude
    
    
End Program bhabha_scattering

subroutine mandlestam(p1, p2, p3, p4, s, t, u)

implicit none 
real(8), dimension(10) :: p1, p2, p3, p4
real(8)                :: s, t, u
integer                :: i


!         s = (p1 + p2)*(p1 + p2)'
!         t = (p1 - p3)*(p1 - p3)'
!         u = (p1 - p2)*(p1 - p2)'

s = 0.0D0
t = 0.0D0
u = 0.0D0

do i=1,4
    s = s + (p1(i) - p3(i)) * (p1(i) - p3(i))
    t = t + (p1(i) + p2(i)) * (p1(i) + p2(i))
    u = u + (p1(i) - p2(i)) * (p1(i) - p2(i))
end do

print *, "s, t, u", s, t, u

end subroutine