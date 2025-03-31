all: ghiep.out hils.out
default: all

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

ghiep.out: ghiep.o
	-${CLINKER} -o ghiep.out ghiep.o ${SLEPC_EPS_LIB}
	${RM} ghiep.o

hils.out: hils.o
	-${CLINKER} -o hils.out hils.o ${SLEPC_EPS_LIB}
	${RM} hils.o
