# ==================== ESTRUCTURA Y EVOLUCION ESTELAR ================
# 
# ========================= Elena Arjona Galvez ======================
# 
# ============================= POLITROPOS ===========================

# Este codigo pretende resolver un sistema de DOS ecuaciones por el metodo de Runge-Kutta 

# =====================================================================

# Manera de ejecutar el codigo: python RungeKutta.py h n mu_e rho_c num nombre_imagen

# donde h es el paso de integracion
# n es el índice politropico
# mu_e es la composicion quimica
# num es el numero al que quieres dividir el paso para calcular el error numerico
# nombre_imagen pues eso, el nombre de la imagen que quieres poner

# IMPORTANTE: no pongas fracciones en el indice politropico, pon numero con decimal, con fracciones puede no funcionar

# ======================================================================

import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.constants as c

plt.ion()


def system(yy0, yy1, zz, n):
    dy0 = yy1
    dy1 = -2.*(yy1/zz) - np.abs(yy0)**n
    return dy0, dy1


def RungeKutta(system, y0_0, y1_0, n, h): 
    y0 = [y0_0]
    y1 = [y1_0]
    z = [h]


    while y0[-1] > 0:
        k1_0, k1_1 = system(y0[-1], y1[-1], z[-1], n)
        k2_0, k2_1 = system(y0[-1] + h*k1_0/2., y1[-1] + h*k1_1/2., z[-1] + h/2., n)
        k3_0, k3_1 = system(y0[-1] + h*k2_0/2., y1[-1] + h*k2_1/2., z[-1] + h/2., n)
        k4_0, k4_1 = system(y0[-1] + h*k3_0, y1[-1] + h*k3_0, z[-1] + h, n)


        y0.append(y0[-1] + h*(k1_0 + 2.*k2_0 + 2.*k3_0 + k4_0)/6.)
        y1.append(y1[-1] + h*(k1_1 + 2.*k2_1 + 2.*k3_1 + k4_1)/6.)
        z.append(z[-1]+h)

    return y0, y1, z





# Para este caso, vamos a tratar, como vienen en los apuntes, un desarrollo en serie para las funciones, tal que: 


def potencias(a1, a2, a3, a4, z0):
    y0_0 = 1. + a1*z0 + a2*z0**2. + a3*z0**3. + a4*z0**4.# Desarrollo en serie hasta la cuarta potencia para w
    y1_0 = a1 + 2.*a2*z0 + 3.*a3*z0**2. + 4.*a4*z0**3.# Desarrollo en serie hasta la cuarta potencia para w'
    return y0_0, y1_0


# Una vez definido todo, procedemos a integrar


n = float(sys.argv[2]) # Indice del politropo, para meterlo desde fuera no puedes meter una fraccion, sino un numero
h = float(sys.argv[1]) # Paso de integracion, prueba 0.01


y0_0, y1_0 = potencias(0., -1./6., 0, n/120., h) # Condiciones iniciales



num_sol_y0,  num_sol_y1, z_num = RungeKutta(system, y0_0, y1_0, n, h) # Resolviendo numericamente

# El radio de la estrella sera cuando la densidad sea cero, pero no podemos calcularlo exactamente, por lo cual vamos a imponer que la densidad sea positiva, entonces el radio sera el correspondiente a z[-2]




# ==============================================================================
# ========================== PARA CALCULAR EL ERROR ============================
# ==============================================================================

# Para calcular el error solo tenemos que comparar nuestra solucion con dos pasos

num = float(sys.argv[5])
h2 = h/num

err0 = [0.]
err1 = [0.]

num_sol_y0_err,  num_sol_y1_err, z_num_err = RungeKutta(system, y0_0, y1_0, n, h2)

for i in range(1,len(z_num)-1):
    ey0 = (num_sol_y0[i-1]-num_sol_y0_err[int(1*num)])**2
    ey1 = (num_sol_y1[i-1]-num_sol_y1_err[int(1*num)])**2

    err0.append(np.sqrt(np.sum(ey0)/(i+1)))
    err1.append(np.sqrt(np.sum(ey1)/(i+1)))

# No existe el error de la solucion perteneciente a la num_sol_y0 negativa, pero esto da igual, porque en ningun caso la usamos


# ==============================================================================
# ======================== SOLUCIONES ANALITICAS ===============================
# ==============================================================================

def analitic(z):
    w0 = 1. - (1./6.)*z**2.
    dw0 = - (2./6.)*z
    w1 = np.sin(z)/z
    dw1 = (np.cos(z)*z-np.sin(z))/z**2
    w5 = 1./np.sqrt(1+(z**2.)/3.)
    return w0, dw0, w1, dw1, w5


# ==============================================================================
# ===================== ERROR ANALITICO ========================================
# ==============================================================================

# Por la estructura de la integral, dado que con un indice politropico de n = 5 nunca llegamos a 0 en la solucion el bucle while nunca parara, por lo cual, analizaremos el error analitico de nuestra integral con respecto a un indice politropico de n = 1, podriamos tambien hacerlo con n = 0

num_sol_y0_ann1,  num_sol_y1_ann1, z_num_ann1 = RungeKutta(system, y0_0, y1_0, 1, h)


w0_1 = np.zeros(len(z_num_ann1))
w1_1 = np.zeros(len(z_num_ann1))
w5_1 = np.zeros(len(z_num_ann1))
dw1_1 = np.zeros(len(z_num_ann1))
dw0_1 = np.zeros(len(z_num_ann1))
for k in range(len(z_num_ann1)):
	w0_1[k], dw0_1[k], w1_1[k], dw1_1[k], w5_1[k] = analitic(z_num_ann1[k])

err0_ann1 = [np.abs(num_sol_y0_ann1[r] - w1_1[r]) for r in range(len(w1_1))]
err1_ann1 = [np.abs(num_sol_y1_ann1[r1] - dw1_1[r1]) for r1 in range(len(dw1_1))]


num_sol_y0_ann0,  num_sol_y1_ann0, z_num_ann0 = RungeKutta(system, y0_0, y1_0, 0, h)


w0_0 = np.zeros(len(z_num_ann0))
w1_0 = np.zeros(len(z_num_ann0))
w5_0 = np.zeros(len(z_num_ann0))
dw1_0 = np.zeros(len(z_num_ann0))
dw0_0 = np.zeros(len(z_num_ann0))
for k0 in range(len(z_num_ann0)):
	w0_0[k0], dw0_0[k0], w1_0[k0], dw1_0[k0], w5_0[k0] = analitic(z_num_ann0[k0])

err0_ann0 = [np.abs(num_sol_y0_ann0[r] - w0_0[r]) for r in range(len(w0_0))]
err1_ann0 = [np.abs(num_sol_y1_ann0[r0] - dw0_0[r0]) for r0 in range(len(dw0_0))]


# =============================================================================
# =========================== SOLUCIONES DIMENSIONALES ========================
# =============================================================================


# Una vez resuelto el sistema de ecuaciones, hay que pasar de soluciones adimensionales a soluciones fisicas


mu_e = float(sys.argv[3])
K = (1./20.)*(3./np.pi)**(2./3.)*(c.h**2./c.m_e)*(1/(mu_e*c.m_u)**(5./3.))
rho_c = float(sys.argv[4])*1e3 

A = np.sqrt((4*np.pi*c.G/(((n+1)**n)*K**n))*((rho_c**(1./n))*(n+1)*K)**(n-1))

R = z_num[-2]/A
P_c = K*rho_c**(1+(1/n))
M = 4*np.pi*rho_c*R**3*((-1/z_num[-2])*num_sol_y1[-2])
errM_num = 4*np.pi*rho_c*R**3*((-1/z_num[-2])*err1[-2]) 
errM_ann1 = 4*np.pi*rho_c*R**3*((-1/z_num[-2])*err1_ann1[-2]) 
errM_ann0 = 4*np.pi*rho_c*R**3*((-1/z_num[-2])*err1_ann0[-2]) 

print('----------------------------------------------------------')
print('Masa total de nuestra estrella:', M, '+- (num)', np.abs(errM_num), '+- (ann n = 1)', np.abs(errM_ann1), '+- (ann n = 0)', np.abs(errM_ann0))
print('Radio de la superficie:', R)
print('Densidad central:', rho_c)
print('Presion central:', P_c)
print('----------------------------------------------------------')

# Recordemos que el radio de la superficie corresponde a una z_num[-2]

r = z_num[:-1]/A
rho = [rho_c*i**n for i in num_sol_y0[:-1]]
err_rho = [rho_c*n*err0[j]*num_sol_y0[j]**(n-1) for j in range(len(num_sol_y0)-1)]
P = [K*j**(1+(1/n)) for j in rho]
m = [4*np.pi*rho_c*r[k]**3*((-1/z_num[k])*num_sol_y1[k]) for k in range(len(z_num)-1)]

rR = r/R
PPc = [p/P_c for p in P]
rhorhoc = [a/rho_c for a in rho]
mM = [mm/M for mm in m]



plt.plot(rR, PPc, label = r'$P/P_{c}$') 
plt.plot(rR, rhorhoc, label = r'$\rho/\rho_{c}$')
plt.plot(rR, mM, label = r'$M/M_{T}$')
plt.xlabel(r'r/$R_{*}$')

if float(sys.argv[4]) == float(1e4): 
  plt.figtext(0.75,0.6,r'$\rho_{c}$='+ r'$10^{4}g/cm^{3}$'+'\n     n = '+str(n))
if float(sys.argv[4]) == 5e5:
  plt.figtext(0.72,0.6,r'$\rho_{c}$='+ r'$5\cdot10^{5}g/cm^{3}$'+'\n       n = '+str(n))
  
if float(sys.argv[4]) == float(1e7): 
  plt.figtext(0.75,0.6,r'$\rho_{c}$='+ r'$10^{7}g/cm^{3}$'+'\n     n = '+str(n))
if float(sys.argv[4]) == float(5e8):
  plt.figtext(0.72,0.6,r'$\rho_{c}$='+ r'$5\cdot10^{8}g/cm^{3}$'+'\n       n = '+str(n))
  
plt.grid(linestyle='--',alpha = 0.6)
plt.legend()
plt.savefig(str(sys.argv[6])+'.png',dpi=300)


plt.figure()
plt.plot(z_num_ann0, num_sol_y0_ann0,'*', label='h = '+str(sys.argv[1])+' and n = 0')
plt.plot(z_num_ann1, num_sol_y0_ann1,'*', label='h = '+str(sys.argv[1])+' and n = 1')
plt.plot(z_num_ann1, w1_1, label='Analitic solution for n = 1')
plt.plot(z_num_ann0, w0_0, label='Analitic solution for n = 0')
plt.grid(linestyle='--',alpha = 0.6)
plt.legend()
plt.xlabel('z')
plt.ylabel('w')
plt.savefig('numericalerror.png', dpi = 300)


figNum = plt.figure()
figNum.add_subplot(1,2,1)
plt.plot(z_num_ann0, np.abs(num_sol_y0_ann0-w0_0),'-', label='h = '+str(sys.argv[1])+' and n = 0')
plt.grid(linestyle='--',alpha = 0.6)
plt.legend()
plt.xlabel('z')
plt.ylabel(r'$w_{numerical}-w_{analitical}$')
plt.tight_layout()
figNum.add_subplot(1,2,2)
plt.plot(z_num_ann1, np.abs(num_sol_y0_ann1-w1_1),'-', label='h = '+str(sys.argv[1])+' and n = 1')
plt.xlabel('z')
plt.ylabel(r'$w_{numerical}-w_{analitical}$')
plt.legend()
plt.grid(linestyle='--',alpha = 0.6)
plt.tight_layout()
plt.savefig('diff.png', dpi = 300)
