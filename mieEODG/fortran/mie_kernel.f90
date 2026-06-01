! mie_kernel.f90 - Fortran 90 Mie scattering kernel for a batch of particles.
!
! Port of mieEODG._mie_kernel_batch (Python/NumPy) to Fortran 90.
! Computes Qext, Qsca, g for a batch of N size parameters at a single
! complex refractive index m.
!
! Sign conventions (match Python _mie_kernel_batch):
!   m   = n - i*k  (negative imaginary for absorbing particles)
!   xi  = psi + i*chi    (Bohren & Huffman outgoing-wave convention)
!   D recurrence start: nmx = max(nstop, ceil(|m*x|)) + 15
!
! Author: port from mieEODG Python / EODG IDL mie.pro

subroutine mie_batch(x_arr, m_re, m_im, n_x, qext, qsca, g)
    implicit none

    integer,  intent(in)                    :: n_x
    real(8),  intent(in),  dimension(n_x)   :: x_arr
    real(8),  intent(in)                    :: m_re
    real(8),  intent(in)                    :: m_im
    real(8),  intent(out), dimension(n_x)   :: qext, qsca, g

    complex(8) :: m, ir
    complex(8) :: y_arr(n_x)

    integer  :: i, n, global_nmx, global_nstop
    integer  :: nstop(n_x), nmx(n_x)
    real(8)  :: x2(n_x), cbrt_x(n_x), dn, rnx

    real(8)    :: psi0(n_x), psi1(n_x), psi(n_x)
    real(8)    :: chi0(n_x), chi1(n_x), chi(n_x)
    complex(8) :: xi0(n_x), xi(n_x)
    complex(8) :: an(n_x), bn(n_x), anm1(n_x), bnm1(n_x)

    complex(8), allocatable :: D(:,:)
    complex(8) :: A1

    real(8)    :: qext_acc(n_x), qsca_acc(n_x), g_acc(n_x)
    real(8)    :: tnp1, tnm1, tmp

    m  = cmplx(m_re, -m_im, kind=8)
    ir = 1.0d0 / m

    do i = 1, n_x
        y_arr(i) = m * x_arr(i)
    end do

    x2     = x_arr ** 2
    cbrt_x = x_arr ** (1.0d0 / 3.0d0)
    do i = 1, n_x
        if (x_arr(i) < 0.02d0) then
            nstop(i) = 2
        else if (x_arr(i) <= 8.0d0) then
            nstop(i) = int(x_arr(i) + 4.00d0*cbrt_x(i) + 2.0d0)
        else if (x_arr(i) < 4200.0d0) then
            nstop(i) = int(x_arr(i) + 4.05d0*cbrt_x(i) + 2.0d0)
        else
            nstop(i) = int(x_arr(i) + 4.00d0*cbrt_x(i) + 2.0d0)
        end if
        nmx(i) = max(nstop(i), int(abs(y_arr(i))) + 1) + 15
    end do

    global_nstop = maxval(nstop)
    global_nmx   = maxval(nmx)

    allocate(D(0:global_nmx, n_x))
    D = cmplx(0.0d0, 0.0d0, kind=8)

    do n = global_nmx - 1, 1, -1
        dn = dble(n + 1)
        do i = 1, n_x
            A1     = cmplx(dn, 0.0d0, kind=8) / y_arr(i)
            D(n,i) = A1 - 1.0d0 / (A1 + D(n+1, i))
        end do
    end do

    ! Init: psi0=cos(x) [psi_{-1}], psi1=sin(x) [psi_0]
    !       chi0=-sin(x)[chi_{-1}], chi1=cos(x) [chi_0]
    !       xi0 = psi_0 + i*chi_0 = sin(x) + i*cos(x)
    do i = 1, n_x
        psi0(i) =  cos(x_arr(i))
        psi1(i) =  sin(x_arr(i))
        chi0(i) = -sin(x_arr(i))
        chi1(i) =  cos(x_arr(i))
        xi0(i)  = cmplx(psi1(i), chi1(i), kind=8)
    end do

    qext_acc = 0.0d0
    qsca_acc = 0.0d0
    g_acc    = 0.0d0
    anm1     = cmplx(0.0d0, 0.0d0, kind=8)
    bnm1     = cmplx(0.0d0, 0.0d0, kind=8)
    tnp1     = 1.0d0

    do n = 1, global_nstop
        dn   = dble(n)
        tnp1 = tnp1 + 2.0d0
        tnm1 = tnp1 - 2.0d0

        do i = 1, n_x
            rnx = dn / x_arr(i)

            psi(i) = tnm1 * psi1(i) / x_arr(i) - psi0(i)
            chi(i) = tnm1 * chi1(i) / x_arr(i) - chi0(i)
            xi(i)  = cmplx(psi(i), chi(i), kind=8)

            ! Skip Mie coefficient computation once the series has converged.
            ! For n > nstop(i) the upward Bessel recurrence is unstable: psi
            ! and chi grow without bound, and Fortran's naive complex division
            ! produces NaN via overflow in intermediate products.  The physical
            ! contribution of these orders is zero.
            !
            ! At exactly n = nstop(i)+1 we must still accumulate the trailing
            ! g cross-term  (2n-1)/(n^2-n) * Re(a_{n-1} conj(b_{n-1}))  which
            ! the main loop would otherwise compute at this iteration using the
            ! last valid anm1/bnm1.  After that we zero anm1/bnm1 so subsequent
            ! iterations (n > nstop+1) contribute nothing more to g_acc.
            if (n > nstop(i)) then
                if (n == nstop(i) + 1 .and. n > 1) then
                    g_acc(i) = g_acc(i) + &
                        tnm1 / (dn*dn - dn) * dble(anm1(i)*conjg(bnm1(i)))
                end if
                anm1(i) = cmplx(0.0d0, 0.0d0, kind=8)
                bnm1(i) = cmplx(0.0d0, 0.0d0, kind=8)
            else
                an(i) = ((D(n,i)*ir + rnx)*psi(i) - psi1(i)) / &
                        ((D(n,i)*ir + rnx)*xi(i)  - xi0(i))
                bn(i) = ((D(n,i)*m  + rnx)*psi(i) - psi1(i)) / &
                        ((D(n,i)*m  + rnx)*xi(i)  - xi0(i))

                qext_acc(i) = qext_acc(i) + tnp1 * dble(an(i) + bn(i))
                qsca_acc(i) = qsca_acc(i) + tnp1 * &
                              (dble(an(i))**2 + dimag(an(i))**2 + &
                               dble(bn(i))**2 + dimag(bn(i))**2)

                if (n > 1) then
                    tmp = dble(anm1(i)*conjg(an(i)) + bnm1(i)*conjg(bn(i)))
                    g_acc(i) = g_acc(i) + (dn*dn - 1.0d0)/dn * tmp + &
                               tnm1/(dn*dn - dn) * dble(anm1(i)*conjg(bnm1(i)))
                end if

                anm1(i) = an(i);    bnm1(i) = bn(i)
            end if

            psi0(i) = psi1(i);  psi1(i) = psi(i)
            chi0(i) = chi1(i);  chi1(i) = chi(i)
            xi0(i)  = cmplx(psi1(i), chi1(i), kind=8)
        end do
    end do

    deallocate(D)

    do i = 1, n_x
        qext(i) = (2.0d0 / x2(i)) * qext_acc(i)
        qsca(i) = (2.0d0 / x2(i)) * qsca_acc(i)
        if (qsca(i) > 0.0d0) then
            g(i) = (4.0d0 / (x2(i) * qsca(i))) * g_acc(i)
        else
            g(i) = 0.0d0
        end if
    end do

end subroutine mie_batch
