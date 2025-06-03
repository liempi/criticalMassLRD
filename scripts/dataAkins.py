import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.io        import ascii
from astropy.table     import vstack,Table
from astropy.constants import G
from astropy.cosmology import FlatLambdaCDM

def criticalMass(radius, mass, age):
	sigma            = np.sqrt(gravitationalConstant*mass/radius)                              # km s ⁻¹
	safronovNumber   = 9.54*(100/sigma)**2                                                     # Assuming R* = 1 R⊙ and M* = 1 M⊙, adimensional
	crossSection     = 16*np.sqrt(np.pi)*(1+safronovNumber)*solarRadius2Parsec**2              # pc²
	criticalMassValue= radius**(7/3)*((4*np.pi)/(3*crossSection*age*np.sqrt(G_custom)))**(2/3) # M⊙
	return criticalMassValue.value

# Define the cosmology as Akins et al. 2024.
cosmo=FlatLambdaCDM(H0=67.66, Om0=0.31)

parentDir='/Users/macbookpro/projects/criticalMassLRD'
# Read data from Kocevski et al. 2024 and Akins et al. 2024

dataAkins24   =Table.read(parentDir+'/data'+'/akins2024.dat',format='ascii.ecsv')

# The sum of boths is 56, there is an extra source that we should remove. 
redshiftAkins24   = np.array(    dataAkins24['z_gal_med'   ])       # Adimensional

stellarMassAkins24Up  =np.array(10**dataAkins24['logMstar_u68']) # M⊙
stellarMassAkins24Low =np.array(10**dataAkins24['logMstar_l68']) # M⊙
stellarMassAkins24Mean=np.array(10**dataAkins24['logMstar_med']) # M⊙

radiusAkins24Up    =np.array(dataAkins24['Reff444_u68'])*u.mas
radiusAkins24Low   =np.array(dataAkins24['Reff444_l68'])*u.mas
radiusF444AkinsMean=np.array(dataAkins24['Reff444_med'])*u.mas # mas


# Convert angular size to radians
angularAngleUp  =radiusAkins24Up    .to(u.rad)
angularAngleLow =radiusAkins24Low   .to(u.rad)
angularAngleMean=radiusF444AkinsMean.to(u.rad)

angularDistanceUp  =cosmo.angular_diameter_distance(redshiftAkins24).to(u.pc)
angularDistanceLow =cosmo.angular_diameter_distance(redshiftAkins24).to(u.pc)
angularDistanceMean=cosmo.angular_diameter_distance(redshiftAkins24).to(u.pc)


# Compute angular diameter distance
radiusAkins24Up  =(angularAngleUp  *angularDistanceUp  ).value
radiusAkins24Low =(angularAngleLow *angularDistanceLow ).value
radiusAkins24Mean=(angularAngleMean*angularDistanceMean).value


logR_AkinsMean  = np.log10(radiusAkins24Mean)
# Approximate to symmetric erros.
logR_AkinsError = 0.5*(np.log10(radiusAkins24Up)-np.log10(radiusAkins24Low))

logM_AkinsMean = np.log10(stellarMassAkins24Mean)
logM_AkinsError= 0.5*(np.log10(stellarMassAkins24Up)-np.log10(stellarMassAkins24Low))


newTable = Table()

newTable['ID']        =dataAkins24['id']
newTable['redshift']  =dataAkins24['z_gal_med']
newTable['logR']      =logR_AkinsMean
newTable['logR_error']=logR_AkinsError
newTable['logM']      =logM_AkinsMean
newTable['logM_error']=logM_AkinsError

ascii.write(newTable,parentDir+'/data/akinsDataMCMC.tex' ,format='latex', overwrite=True)


"""
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