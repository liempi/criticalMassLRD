import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.io        import ascii
from astropy.table     import vstack,Table
from astropy.constants import G
from astropy.cosmology import FlatLambdaCDM

gravitationalConstant = 4.3009172706e-3 # pc M⊙⁻¹ (km s⁻¹)²
solarRadius2Parsec    = 2.2567e-8       # 1 R⊙ -> pc
G_custom = G.to(u.pc**3 / (u.M_sun * u.Gyr**2)).value

inches=2.54
plt.rcParams['path.simplify_threshold'] = 0.111111111111
plt.rcParams['agg.path.chunksize'     ] = 10000
plt.rcParams['figure.figsize'         ] = [8.5/inches, 8.5/inches]
plt.rcParams['xtick.labelsize'        ] = 10
plt.rcParams['ytick.labelsize'        ] = 10
plt.rcParams['legend.fontsize'        ] = 9
plt.rcParams['legend.handletextpad'   ] = 0.1  # Set the default padding
plt.rcParams["legend.borderpad"       ] = 0.2
plt.rcParams['legend.handleheight'    ] = 0.5  # the height of the legend handle
plt.rcParams['legend.handlelength'    ] = 0.5  # the height of the legend handle
plt.rcParams['legend.borderaxespad'   ] = 0.1
plt.rcParams['legend.columnspacing'   ] = 0.5
plt.rcParams['font.family'            ] = 'serif'
plt.rcParams['font.serif'             ] = 'dejavuserif'
plt.rcParams['text.usetex'            ] = True
plt.rcParams['mathtext.fontset'       ] = 'dejavuserif'
plt.rcParams['lines.markersize'       ] = 2


def song2016(Muv, z):
    stellarMass = np.zeros_like(Muv, dtype=float)
    mask_z4 = (z == 4)
    mask_z5 = (z == 5)
    mask_z6 = (z == 6)
    mask_z7 = (z == 7)
    mask_z8 = (z == 8)

    stellarMass[mask_z4] = 10**(-0.54 * Muv[mask_z4] - 1.70)
    stellarMass[mask_z5] = 10**(-0.50 * Muv[mask_z5] - 0.90)
    stellarMass[mask_z6] = 10**(-0.50 * Muv[mask_z6] - 1.04)
    stellarMass[mask_z7] = 10**(-0.50 * Muv[mask_z7] - 1.20)
    stellarMass[mask_z8] = 10**(-0.50 * Muv[mask_z8] - 1.50)
    
    return stellarMass

def stefanon2021(Muv,z):
    stellarMass = np.zeros_like(Muv, dtype=float)
    mask_z3 = (z == 3)
    mask_z4 = (z == 4)
    mask_z5 = (z == 5)

    mask_z6 = (z == 6)
    mask_z7 = (z == 7)
    mask_z8 = (z == 8)
    mask_z9 = (z == 9)
    mask_z10= (z ==10)
    mask_z11= (z ==11)

    # Lee (2012)
    stellarMass[mask_z3 ] = 10**(-0.41 * Muv[mask_z3 ] + 1.200)
    # Song (2016)    
    stellarMass[mask_z4 ] = 10**(-0.54 * Muv[mask_z4 ] - 1.700)
    stellarMass[mask_z5 ] = 10**(-0.50 * Muv[mask_z5 ] - 0.900)
    # Stefanon (2021)
    stellarMass[mask_z6 ] = 10**(-0.57 * Muv[mask_z6 ] - 1.064)
    stellarMass[mask_z7 ] = 10**(-0.49 * Muv[mask_z7 ] - 1.056)
    stellarMass[mask_z8 ] = 10**(-0.49 * Muv[mask_z8 ] - 1.047)
    stellarMass[mask_z9 ] = 10**(-0.46 * Muv[mask_z9 ] - 1.028)
    stellarMass[mask_z10] = 10**(-0.46 * Muv[mask_z10] - 1.023)
    stellarMass[mask_z11] = 10**(-0.46 * Muv[mask_z11] - 1.023)

    return stellarMass

def kawamata2018(Muv,z):
	L0           = 3.91e10 # M_UV= -21 in L⊙ units 
	uvLuminosity = 10**(0.4*(5.48-Muv))
	radius = np.zeros_like(Muv, dtype=float)
	mask_z3 = (z == 3)
	mask_z4 = (z == 4)
	mask_z5 = (z == 5)
	mask_z6 = (z == 6)
	mask_z7 = (z == 7)
	mask_z8 = (z == 8)
	mask_z9 = (z == 9)
	mask_z10= (z ==10)
	mask_z11= (z ==11)
	    # Shibuya et al. (2015)
	radius[mask_z3 ] = 1.30e3*(uvLuminosity[mask_z3 ]/L0)**0.27
	radius[mask_z4 ] = 1.00e3*(uvLuminosity[mask_z4 ]/L0)**0.27
	radius[mask_z5 ] = 0.80e3*(uvLuminosity[mask_z5 ]/L0)**0.27
		# Kawamata et al. (2018)
	radius[mask_z6 ] = 0.94e3*(uvLuminosity[mask_z6 ]/L0)**0.46
	radius[mask_z7 ] = 0.94e3*(uvLuminosity[mask_z7 ]/L0)**0.46
	radius[mask_z8 ] = 0.81e3*(uvLuminosity[mask_z8 ]/L0)**0.38
	# Yang et al. (2020)
	radius[mask_z9 ] = 1.20e3*(uvLuminosity[mask_z9 ]/L0)**0.56
	radius[mask_z10] = 0.44e3*(uvLuminosity[mask_z10]/L0)**0.20
	radius[mask_z11] = 0.44e3*(uvLuminosity[mask_z11]/L0)**0.20
	return radius

def criticalMass(radius, mass, age):
	sigma            = np.sqrt(gravitationalConstant*mass/radius)                              # km s ⁻¹
	safronovNumber   = 9.54*(100/sigma)**2                                                     # Assuming R* = 1 R⊙ and M* = 1 M⊙, adimensional
	crossSection     = 16*np.sqrt(np.pi)*(1+safronovNumber)*solarRadius2Parsec**2              # pc²
	criticalMassValue= radius**(7/3)*((4*np.pi)/(3*crossSection*age*np.sqrt(G_custom)))**(2/3) # M⊙
	return criticalMassValue.value

# Define the cosmology as Akins et al. 2024.
cosmo=FlatLambdaCDM(H0=67.66, Om0=0.31)

parentDir='/Users/macbookpro/projects/criticalMassLRD'
figuresDir=parentDir+'/figures'
# Read data from Kocevski et al. 2024 and Akins et al. 2024

dataAkins24   =Table.read(parentDir+'/data'+'/akins2024.dat',format='ascii.ecsv')
"""
jadesData = dataKocevski24[(dataKocevski24['ID']=='JADES' )]
ngdeepData= dataKocevski24[(dataKocevski24['ID']=='NGDEEP')]

# Combine the two tables
combinedData = vstack([jadesData, ngdeepData])
"""
# Write to a new file
"""

custom_names = ['Survey', 'ID', 'RA', 'Dec', 'zbest', 'zflag',
                'beta_UV', 'beta_UV_err', 'beta_opt', 'beta_opt_err',
                'mag444', 'm_uv', 'M_UV']

dataKocevski24=ascii.read(parentDir+'/data/kocevski24.dat',names=custom_names)

jadesData = dataKocevski24[(dataKocevski24['Survey']=='JADES' )]
ngdeepData= dataKocevski24[(dataKocevski24['Survey']=='NGDEEP')]
"""

dataKocevski24         =ascii.read(parentDir+'/data'+'/kocevski24_combined.dat')
mask                   =(dataKocevski24['ID']!= 21925 ) #Remove the source which is also removed in Sacchi et al. 2025
dataKocevski24         =dataKocevski24[mask]
uvMagnitudeKocevski24  =dataKocevski24['M_UV']
redshiftKocevski24     =dataKocevski24['zbest']
redshiftRoundKocevski24=np.round(dataKocevski24['zbest'])
stellarMassKocevski24  =stefanon2021(uvMagnitudeKocevski24,redshiftRoundKocevski24)
radiusKocevski24       =kawamata2018(uvMagnitudeKocevski24,redshiftRoundKocevski24)
ageKocevski24          =cosmo.age(redshiftKocevski24)
criticalMassKocevski24 =criticalMass(radiusKocevski24,stellarMassKocevski24,ageKocevski24)


# The sum of boths is 56, there is an extra source that we should remove. 
redshiftAkins24   = np.array(    dataAkins24['z_gal_med'   ])       # Adimensional
radiusF444Akins   = np.array(    dataAkins24['Reff444_med' ])*u.mas # mas
stellarMassAkins24= np.array(10**dataAkins24['logMstar_med'])       # M⊙

stellarMassAkins24Low=np.array(10**dataAkins24['logMstar_l68'])       # M⊙
stellarMassAkins24Up =np.array(10**dataAkins24['logMstar_u68'])       # M⊙

radiusAkins24Low = np.array(dataAkins24['Reff444_l68' ])*u.mas
radiusAkins24Up  = np.array(dataAkins24['Reff444_u68' ])*u.mas

angularAngleLow= radiusAkins24Low.to(u.rad)
angularAngleUp= radiusAkins24Up.to(u.rad)
angularDistanceLow =cosmo.angular_diameter_distance(redshiftAkins24).to(u.pc)
# Convert angular size to radians
angularAngle    = radiusF444Akins.to(u.rad)
# Compute angular diameter distance
angularDistance = cosmo.angular_diameter_distance(redshiftAkins24).to(u.pc)
radiusAkins24   = (angularAngle*angularDistance).value
# Estimate the age of the universe in Gyr
ageAkins24      = cosmo.age(redshiftAkins24)

criticalMassAkins24 = criticalMass(radiusAkins24,stellarMassAkins24,ageAkins24)

newTable = Table()

newTable['ID']          =dataAkins24['id']
newTable['Redshift']    =dataAkins24['z_gal_med'   ]
newTable['Radius_med']  =np.round(radiusF444Akins,2)
newTable['age']         =np.round(ageAkins24,2)
# Format numbers in LaTeX scientific notation
formatted_mass = [f"${val:.2e}$" for val in criticalMassAkins24 ]
newTable['criticalMass'] = formatted_mass
formatted_mass = [f"${val:.2e}$" for val in stellarMassAkins24]
newTable['stellarMass'] = formatted_mass
ascii.write(newTable,'newDataAkins.tex' ,format='latex', overwrite=True)
"""
plt.hist(uvMagnitudeKocevski24, color='blue'   , label='Kocevski et al. (2024)')

plt.xlabel(r'M$_{\rm UV}$')
plt.ylabel('counts')

plt.tight_layout()
plt.savefig(figuresDir+'/absoluteMagnitude.jpg',dpi=300)
plt.close()

plt.hist(np.log10(radiusAkins24)   , color='darkred', label='Akins et al. (2024)')
plt.hist(np.log10(radiusKocevski24), color='blue'   , label='Kocevski et al. (2024)')

plt.yscale('log')
plt.legend()
plt.xlabel(r'log(radius/pc)')
plt.ylabel('counts')

plt.tight_layout()
plt.savefig(figuresDir+'/radius.jpg',dpi=300)
plt.close()


plt.scatter(redshiftAkins24   , stellarMassAkins24   , s=5, color='darkred', label='Akins et al. (2024)')
plt.scatter(redshiftKocevski24, stellarMassKocevski24, s=5, color='blue'   , label='Kocevski et al. (2024)')

plt.legend()
plt.yscale('log')

plt.ylabel(r'$M_{\rm gal}~[{\rm M_\odot}]$')
plt.xlabel('redshift')

plt.tight_layout()
plt.savefig(figuresDir+'/redshift_massStellar.jpg',dpi=300)
plt.close()

plt.scatter(stellarMassAkins24   , radiusAkins24   , s=5, color='darkred', label='Akins et al. (2024)')
plt.scatter(stellarMassKocevski24, radiusKocevski24, s=5, color='blue'   , label='Kocevski et al. (2024)')

plt.legend()
plt.yscale('log')
plt.xscale('log')

plt.xlabel(r'$M_{\rm gal}~[{\rm M_\odot}]$')
plt.ylabel(r'radius$~[{\rm pc}]$')

plt.tight_layout()
plt.savefig(figuresDir+'/radius_massStellar.jpg',dpi=300)
plt.close()

plt.scatter(ageAkins24   , stellarMassAkins24   , s=5, color='darkred', label='Akins et al. (2024)'   )
plt.scatter(ageKocevski24, stellarMassKocevski24, s=5, color='blue'   , label='Kocevski et al. (2024)')

plt.legend()
plt.yscale('log')

plt.ylabel(r'$M_{\rm gal}~[{\rm M_\odot}]$')
plt.xlabel(r'age$~[{\rm Gyr}]$')

plt.tight_layout()
plt.savefig(figuresDir+'/age_massStellar.jpg',dpi=300)
plt.close()

plt.hist(np.log10(stellarMassAkins24   /criticalMassAkins24   ), color='darkred',label='Akins et al. (2024)'   )
plt.hist(np.log10(stellarMassKocevski24/criticalMassKocevski24), color='blue'   ,label='Kocevski et al. (2024)')

plt.legend()

plt.yscale('log')
plt.xlabel(r'$\log{\left(\frac{M_{\rm gal}}{M_{\rm crit}}\right)}$')
plt.ylabel('counts')
plt.tight_layout()
plt.savefig(figuresDir+'/massStellarOverCriticalMass.jpg',dpi=300)
"""