         SUBROUTINE XPOT(RI,IPOT,POTC)
         IMPLICIT REAL*8(A-H,O-Z)
         DIMENSION C2(20),F2(20)
         dimension xfatt(0:16),xc(0:16)
C
         IF(IPOT.EQ.1) THEN
         D=7.449D0
         C2(3)=1.461D0
         C2(4)=14.11D0
         C2(5)=183.5D0
         BETA=1.3443D0
         ER1=3.5D0/BETA-1.D0
         ER2=2.0D0*BETA
         NMAX=12
C
         DO N=6,NMAX
         C2(N)=(C2(N-1)/C2(N-2))**3*C2(N-3)
         END DO
C
         XI=RI
         XD=1.D0/XI
         VH1=D*(XI**ER1)*DEXP(-ER2*XI)
C
         BR=ER2-ER1*XD
         BRX=BR*XI
         SUMV=0.D0
          DO N=3,NMAX
          NN=2*N
          SUM=0.D0
           DO K=0,NN
           SUM=SUM+(BRX**K)/DGAMMA(K+1.D0)
           END DO
          F2N=1.D0-DEXP(-BRX)*SUM
          if(dabs(f2n).lt.1.d-12) then
           aaa=(brx**(nn+1))/dgamma(nn+2.d0)
           f2n=dexp(-brx)*aaa
          end if
          F2(N)=F2N
          SUMV=SUMV-F2(N)*C2(N)*(XD**NN)
          END DO
         VH2=SUMV
C
         POTC=(VH1+VH2)/3.1669D-6
C        IF(RI.LT.0.5D0)POTC=VH1
         END IF
C
c   The dispersive part only
         IF(IPOT.EQ.10) THEN
         D=7.449D0
         C2(3)=1.461D0
         C2(4)=14.11D0
         C2(5)=183.5D0
         BETA=1.3443D0
         ER1=3.5D0/BETA-1.D0
         ER2=2.0D0*BETA
         NMAX=12
C
         DO N=6,NMAX
         C2(N)=(C2(N-1)/C2(N-2))**3*C2(N-3)
         END DO
C
         XI=RI
         XD=1.D0/XI
         VH1=D*(XI**ER1)*DEXP(-ER2*XI)
C
         BR=ER2-ER1*XD
         BRX=BR*XI
         SUMV=0.D0
          DO N=3,NMAX
          NN=2*N
          SUM=0.D0
           DO K=0,NN
           SUM=SUM+(BRX**K)/DGAMMA(K+1.D0)
           END DO
          F2N=1.D0-DEXP(-BRX)*SUM
          if(dabs(f2n).lt.1.d-12) then
           aaa=(brx**(nn+1))/dgamma(nn+2.d0)
           f2n=dexp(-brx)*aaa
          end if
          F2(N)=F2N
          SUMV=SUMV-F2(N)*C2(N)*(XD**NN)
          END DO
         VH2=SUMV
C
         POTC=(VH2)/3.1669D-6
C        IF(RI.LT.0.5D0)POTC=VH1
         END IF

C
         IF(IPOT.EQ.2) THEN
      R=RI
C     POTC=(1458.0470D0*DEXP(-3.11D0*R)-578.0890D0*DEXP(-1.55D0*R))/R
C     POTC=(1438.4812D0*DEXP(-3.11D0*R)-570.3316D0*DEXP(-1.55D0*R))/R
C     POTC=(1438.7200D0*DEXP(-3.11D0*R)-626.8850D0*DEXP(-1.55D0*R))/R
      r0=10.03d0
      ua0=.09970089730807577268d0
      ww0=1.227d0
      sa0=(r*ua0)**2
      potc=-ww0*dexp(-sa0)
         END IF
C
         IF(IPOT.EQ.3) THEN
         PI=4.D0*DATAN(1.D0)
C        E0=10.97D0*3.1669D-6
         E0=10.97D0
         A0=1.89635353D5
         AF=10.70203539D0
         BF=1.90740649D0
         C6=1.34687065D0
         C8=0.41308398D0
         C10=0.17060159D0
         D0=1.4088D0
         RMN=5.6115D0
         B0=0.0026D0
         X0=1.003535949D0
         X1=1.454790369D0
         X=RI/RMN
C
          IF(X.LT.D0) THEN
           XXX=(D0/X-1.D0)**2
           FX=DEXP(-XXX)
          ELSE
           FX=1.D0
          END IF
         X6=X**6
         X8=X**8
         X10=X**10
         CXX=C6/X6+C8/X8+C10/X10
         AX=-AF*X-BF*X**2
          IF(X.LT.X1.AND.X.GT.X0) THEN
           PPP=2*PI*(X-X0)/(X1-X0)-0.5D0*PI
           UA0=DSIN(PPP)+1.D0
          ELSE
           UA0=0.D0
          END IF
C
         POTC=E0*(A0*DEXP(AX)-CXX*FX+B0*UA0)
C
         END IF
C
 	 IF(IPOT.EQ.4) THEN
	epsu=10.948D0*3.1669D-6
	A=1.8443101d5
	alfa=10.43329537d0
	beta=2.27965105d0
	C6=1.36745214d0
	C8=0.42123807d0
	C10=0.17473318d0
	D=1.4826d0
	rmina=2.963d0
	rminu=5.59933480733979628474d0
	urminu=.1785926425919676004d0
	urmina=.3374957813027337158d0
	pi=.31415926535897931160d1
C
	urmin=urminu
       eps=epsu
C
	xx=ri
	if (xx.eq.0.d0) xx=0.00000000000000000000000000001d0
C	
	x1=xx*urmin
      x2=x1**2
	ux2=1.d0/x2
	ux1=1.d0/x1
C    
	 if (x1.ge.D) then
        dx=1.d0
      else
	 dd=(D*ux1-1.D0)**2
	 dx=dexp(-dd)
      end if
C
	pae=alfa*x1+beta*x2
		 ux6=ux2**3
		 ux8=ux6*ux2
		 ux10=ux8*ux2
C	
		 yy1=eps*(A*dexp(-pae))
		 yy2=-eps*((C6*ux6+C8*ux8+C10*ux10)*dx)
C
	potc=yy1+yy2
C
	 END IF
C
 	 IF(IPOT.EQ.5) THEN
C
       A=2.07436426d6
       alfa=1.88648251d0
       beta=-6.20013490d-2
       delta=1.94861295d0
       C6=1.4609778d0
       C8=1.4117855d1
       C10=1.8369125d2
       C12=3.265d3
       C14=7.644d4
       C16=2.275d6
C
       urminu=.1785926425919676004d0
       urmin=urminu
       xr=ri
        if (xr.eq.0.d0) xr=0.001d0
        do i=0,16
           if (i.eq.0) then
             xfatt(0)=1.d0
           else
              xfatt(i)=xfatt(i-1)*dfloat(i)
           end if
           xc(i)=0.d0
        end do      
        xc(6)=C6
        xc(8)=C8
        xc(10)=C10
        xc(12)=C12
        xc(14)=C14
        xc(16)=C16
        xvbm=A*dexp(-alfa*xr+beta*xr*xr)
        y=0.d0
        b=delta
        xdr=xr*delta
        xsum2=0.d0
        do n=3,8
           m=2*n
           xsum3=0.d0
           do i=0,m
           xsum3=xsum3+(xdr**i)/xfatt(i) 
           end do   
           xsum3=1.d0-xsum3*dexp(-b*xr)
           xsum2=xsum2+xsum3*xc(m)/(xr**m)
        end do
        y=(xvbm*3.1669D-6-xsum2)
c       y=(xvbm-xsum2)*3.1669D-6
      POTC=y
      END IF
C
      IF(IPOT.EQ.6) THEN
      XX1=RI
      CALL SAPT1(xx1,f10he)
      POTC=f10he*3.1669D-6
      END IF
C
      IF(IPOT.EQ.7) THEN
      XX1=RI
      CALL SAPT2(xx1,f10he)
      POTC=f10he*3.1669D-6
      END IF
C
      IF(IPOT.EQ.8) THEN
      XX1=RI
      A=1726.743D4
      ALF=5.04D0
      C6=1.123255D4
      D0=0.7455D0
       IF(XX1.LT.D0) THEN
        XXX=(D0/XX1-1.D0)**2
        FX=DEXP(-XXX)
       ELSE
        FX=1.D0
       END IF
      POTC=A*DEXP(-XX1*ALF)-FX*C6/(XX1**6)
      END IF
C
C      IF(IPOT.EQ.9) THEN
C      XX1=RI
C      CALL POTHEHE(xx1,f10he)
C      POTC=f10he
C      END IF
C
      RETURN
      END 
C
      SUBROUTINE SAPT1(xx,f10he)

      implicit real*8 (a-g,o-z)

      dimension xfatt(0:19),ufatt(0:19),xc(0:16),xc2(0:19),xsm3(0:16)

       DATA epsu/0.023106651d0/
       DATA epsa/0.082515414d0/

       DATA A/2.07436426d6/
       DATA A1/6.5693041d1/
       DATA A2/3.1576621d5/
       DATA alfa/1.88648251d0/
       DATA beta/-6.20013490d-2/
       DATA delta/1.94861295d0/
       DATA C6/1.4609778d0/
       DATA C8/1.4117855d1/
       DATA C10/1.8369125d2/
       DATA C12/3.265d3/
       DATA C14/7.644d4/
       DATA C16/2.275d6/

       DATA P11/1.d0/
       DATA P12/-9.431089d-8/
       DATA P13/-2.408169d-5/
       DATA P14/3.576804d-7/
       DATA P15/-4.701832d-9/

       DATA P21/-1.62343d-3/
       DATA P22/2.22097d-3/
       DATA P23/-1.17323d-3/
       DATA P24/3.00012d-4/
       DATA P25/-1.05512d-5/

       DATA P31/8.82506d-2/
       DATA P32/3.81846d-2/
       DATA P33/-1.72421d-3/
       DATA P34/4.74897d-7/
       DATA P35/3.0445706d-3/

       DATA P41/1.488897d0/
       DATA P42/-2.123572d0/
       DATA P43/1.043994d0/
       DATA P44/-1.898459d-1/
       DATA P45/6.479867d-3/
 
       DATA P51/6.184108d-6/
       DATA P52/3.283043d2/
       DATA P53/1.367922d3/
       DATA P54/-4.464489d7/
       DATA P55/1.365003d10/

       DATA P61/-1.107002d-7/
       DATA P62/3.284717d2/
       DATA P63/-9.819846d2/
       DATA P64/-1.953816d7/
       DATA P65/-1.079712d11/

       DATA D/1.4826d0/
       DATA rmina/2.963d0/
       DATA rminu/5.59933480733979628474d0/
       DATA urminu/.1785926425919676004d0/
       DATA urmina/.3374957813027337158d0/
       DATA pi/.31415926535897931160d1/

       DATA xfatt/1.d0,1.d0,2.d0,6.d0,24.d0,120.d0,720.d0,5040.d0,
     y 40320.d0,362880.d0,3628800.d0,39916800.d0, 479001600.d0,
     y  6.2270208d9,8.71782912d10,1.30767437d12,2.09227899d13,
     y 0.3556874280960d15,0.6402373705728d16,0.1216451004088d18/


       DATA ufatt/1.d0,1.d0,0.5d0,0.1666666666667d0,0.4166666666667d-1,
     y 0.8333333333333d-2,0.1388888888889d-2,0.1984126984127d-3,
     y 0.2480158730159d-4,0.2755731922399d-5,0.2755731922399d-6,
     y 0.2505210838544d-7,0.2087675698787d-8,0.1605904383682d-9,
     y 0.1147074559773d-10,0.7647163720124d-12,0.4779477329646d-13
     y ,0.2811457254346d-14,0.1561920696859d-15,0.8220635246624d-17/




              urmin=urminu
              eps=epsu
              xr=xx

        if (xr.le.0.4d0) xr=0.4d0

c        do i=0,16
c           if (i.eq.0) then
c             xfatt(0)=1.d0
c           else
c              xfatt(i)=xfatt(i-1)*dfloat(i)
c           end if
c           xc(i)=0.d0
c           write(27,1055)xfatt(i)*ufatt(i),ufatt(i)
c        end do      

        xc(6)=C6
        xc(8)=C8
        xc(10)=C10
        xc(12)=C12
        xc(14)=C14
        xc(16)=C16

        xsqr=dsqrt(xr)
        x2r=xr*xr
        uxr=1.d0/xr
        ux2r=1.d0/x2r

        if (xr.le.10.d0)
     y xf=p11+p12*xr+p13*x2r+p14*(xr**3)+p15*x2r*x2r 

        if (xr.le.100.d0.and.xr.gt.10.d0)
     y xf=1.d0-p21-p22*xsqr-p23*xr-p24*xr*xsqr-p25*x2r 

        if (xr.le.200.d0.and.xr.gt.100.d0)
     y xf=(1.d0+p31+p32*xsqr+p33*xr+p34*x2r)/(1.2d0+0.8d0*p35*xr) 

        if (xr.le.1000.d0.and.xr.gt.200.d0) then
       xf=(p41*(xr**0.4d0)+p42*xsqr+p43*(xr**0.6d0)+p44*
     y (xr**0.7d0)+p45*(xr**0.8d0))
       xf=dlog(xf*xr)
       end if

        if (xr.le.10000.d0.and.xr.gt.1000.d0)
     y xf=p51+p52*uxr+p53*ux2r+p54*uxr*ux2r+p55*ux2r*ux2r

        if (xr.le.100000.d0.and.xr.gt.10000.d0)
     y xf=p61+p62*uxr+p63*ux2r+p64*uxr*ux2r+p65*ux2r*ux2r


        xvbm=A*dexp(-alfa*xr+beta*x2r)


        xdr=xr*delta

        xesp=dexp(-xdr)


c        write(23,*)xdr,delta,xr,xesp

        do k=0,19
          xc2(k)=(xdr**k)*ufatt(k)
        end do  

        xsum1=0.d0
        do k=0,6
          xsum1=xsum1+xc2(k)
        end do  

        xvret1=(1.d0-xsum1*xesp)*C6*xf*ux2r*ux2r*ux2r

        xsum2=0.d0

        xsm3(0)=1.d0
           do k=1,16
               axa=xc2(k)
c              axa=(xdr**k)/xfatt(k)
             xsm3(k)=xsm3(k-1)+axa
           end do


        do n=4,8
           m=2*n
           xsum3=0.d0
           xprobl=1.d0-xsm3(m)*xesp
c           xprobl2=1.d0-xsm3(m)*xesp

           if (xr.le.0.5d0) xprobl=xc2(m+1)-xc2(m+2)*dfloat(m+1)+0.5d0
     y *xc2(m+3)*dfloat((m+1)*(m+2))

           xsum2=xsum2+xprobl*xc(m)/(xr**m)
c        write(24,1055)xprobl,0.5d0*xc2(m+3)*dfloat((m+1)*(m+2)),xprobl2
c        write(24,*)m,xr
       end do

        y=(xvbm-A2*(xsum2+xvret1))

c       write(25,*)xvbm,-A2*xsum2,-A2*xvret1,xx

      f10he=y

 1055 format(1D20.13,3x,1D20.13,3x,1D20.13)

      RETURN
      END

      SUBROUTINE SAPT2(xx,f12he)

      implicit real*8 (a-g,o-z)

      dimension xfatt(0:19),ufatt(0:19),xc(0:16),xc2(0:19),xsm3(0:16)

       DATA epsu/0.023106651d0/
       DATA epsa/0.082515414d0/

       DATA A/2.07436426d6/
       DATA A1/6.5693041d1/
       DATA A2/3.1576621d5/
       DATA alfa/1.88648251d0/
       DATA beta/-6.20013490d-2/
       DATA delta/1.94861295d0/
       DATA C6/1.4609778d0/
       DATA C8/1.4117855d1/
       DATA C10/1.8369125d2/
       DATA C12/3.265d3/
       DATA C14/7.644d4/
       DATA C16/2.275d6/

       DATA P11/1.d0/
       DATA P12/9.860029d-1/
       DATA P13/5.942027d-3/
       DATA P14/-7.924833d-4/
       DATA P15/3.172548d-5/

       DATA P21/-1.62343d-3/
       DATA P22/2.22097d-3/
       DATA P23/-1.17323d-3/
       DATA P24/3.00012d-4/
       DATA P25/-1.05512d-5/

       DATA P31/8.82506d-2/
       DATA P32/3.81846d-2/
       DATA P33/-1.72421d-3/
       DATA P34/4.74897d-7/
       DATA P35/3.0445706d-3/

       DATA P41/1.488897d0/
       DATA P42/-2.123572d0/
       DATA P43/1.043994d0/
       DATA P44/-1.898459d-1/
       DATA P45/6.479867d-3/

       DATA P51/6.184108d-6/
       DATA P52/3.283043d2/
       DATA P53/1.367922d3/
       DATA P54/-4.464489d7/
       DATA P55/1.365003d10/

       DATA P61/-1.107002d-7/
       DATA P62/3.284717d2/
       DATA P63/-9.819846d2/
       DATA P64/-1.953816d7/
       DATA P65/-1.079712d11/

       DATA D/1.4826d0/
       DATA rmina/2.963d0/
       DATA rminu/5.59933480733979628474d0/
       DATA urminu/.1785926425919676004d0/
       DATA urmina/.3374957813027337158d0/
       DATA pi/.31415926535897931160d1/

       DATA xfatt/1.d0,1.d0,2.d0,6.d0,24.d0,120.d0,720.d0,5040.d0,
     y 40320.d0,362880.d0,3628800.d0,39916800.d0, 479001600.d0,
     y  6.2270208d9,8.71782912d10,1.30767437d12,2.09227899d13,
     y 0.3556874280960d15,0.6402373705728d16,0.1216451004088d18/


       DATA ufatt/1.d0,1.d0,0.5d0,0.1666666666667d0,0.4166666666667d-1,
     y 0.8333333333333d-2,0.1388888888889d-2,0.1984126984127d-3,
     y 0.2480158730159d-4,0.2755731922399d-5,0.2755731922399d-6,
     y 0.2505210838544d-7,0.2087675698787d-8,0.1605904383682d-9,
     y 0.1147074559773d-10,0.7647163720124d-12,0.4779477329646d-13
     y ,0.2811457254346d-14,0.1561920696859d-15,0.8220635246624d-17/




              urmin=urminu
              eps=epsu
              xr=xx

        if (xr.le.0.4d0) xr=0.4d0

        do i=0,16
c           if (i.eq.0) then
c             xfatt(0)=1.d0
c           else
c              xfatt(i)=xfatt(i-1)*dfloat(i)
c           end if
           xc(i)=0.d0
c           write(27,1055)xfatt(i)*ufatt(i),ufatt(i)
        end do      

        xc(6)=C6
        xc(8)=C8
        xc(10)=C10
        xc(12)=C12
        xc(14)=C14
        xc(16)=C16

        xsqr=dsqrt(xr)
        x2r=xr*xr
        uxr=1.d0/xr
        ux2r=1.d0/x2r

        if (xr.le.5.7d0)  xf=1.d0
        if (xr.gt.5.7d0.and.xr.le.10.d0)
     y xf=p12+p13*xr+p14*x2r+p15*(xr**3)

        if (xr.le.100.d0.and.xr.gt.10.d0)
     y xf=1.d0-p21-p22*xsqr-p23*xr-p24*xr*xsqr-p25*x2r 

        if (xr.le.200.d0.and.xr.gt.100.d0)
     y xf=(1.d0+p31+p32*xsqr+p33*xr+p34*x2r)/(1.2d0+0.8d0*p35*xr) 

        if (xr.le.1000.d0.and.xr.gt.200.d0) then
       xf=(p41*(xr**0.4d0)+p42*xsqr+p43*(xr**0.6d0)+p44*
     y (xr**0.7d0)+p45*(xr**0.8d0))
       xf=dlog(xf*xr)
       end if

        if (xr.le.10000.d0.and.xr.gt.1000.d0)
     y xf=p51+p52*uxr+p53*ux2r+p54*uxr*ux2r+p55*ux2r*ux2r

        if (xr.le.100000.d0.and.xr.gt.10000.d0)
     y xf=p61+p62*uxr+p63*ux2r+p64*uxr*ux2r+p65*ux2r*ux2r


        xvbm=A*dexp(-alfa*xr+beta*x2r)


        xdr=xr*delta

        xesp=dexp(-xdr)


c        write(23,*)xdr,delta,xr,xesp

        do k=0,19
          xc2(k)=(xdr**k)*ufatt(k)
        end do  

        xsum1=0.d0
        do k=0,6
          xsum1=xsum1+xc2(k)
        end do  

        xvret1=(1.d0-xsum1*xesp)*C6*xf*ux2r*ux2r*ux2r

        xsum2=0.d0

        xsm3(0)=1.d0
           do k=1,16
               axa=xc2(k)
c              axa=(xdr**k)/xfatt(k)
             xsm3(k)=xsm3(k-1)+axa
           end do


        do n=4,8
           m=2*n
           xsum3=0.d0
           xprobl=1.d0-xsm3(m)*xesp
c           xprobl2=1.d0-xsm3(m)*xesp

           if (xr.le.0.5d0) xprobl=xc2(m+1)-xc2(m+2)*dfloat(m+1)+0.5d0
     y *xc2(m+3)*dfloat((m+1)*(m+2))

           xsum2=xsum2+xprobl*xc(m)/(xr**m)
c        write(24,1055)xprobl,0.5d0*xc2(m+3)*dfloat((m+1)*(m+2)),xprobl2
c        write(24,*)m,xr
       end do

        y=(xvbm-A2*(xsum2+xvret1))

c       write(25,*)xvbm,-A2*xsum2,-A2*xvret1,xx

      f12he=y

 1055 format(1D20.13,3x,1D20.13,3x,1D20.13)

      RETURN
      END
C
C
        SUBROUTINE AZIZ(R,POTC)
        implicit real*8(a-h,o-z)
C 
        eps=10.948d0
C       rm=2.963d0
        rm=5.5993348d0
        aa=1.8443101d5
        alfa=10.43329537d0
        beta=-2.27965105d0
        d=1.4826d0
        c6=1.36745214d0
        c8=0.42123807d0
        c10=0.17473318d0
        x=r/rm
        if(x.lt.d) then
        step=dexp(-(d/x-1)**2)
        else
        step=1
        endif
        if(x.gt.0) then
        sum=c6/x**6+c8/x**8+c10/x**10
        else
        sum=1
        endif
        potc=eps*(aa*exp(-alfa*x+beta*x*x)-step*sum)
        return
        end
C
C
