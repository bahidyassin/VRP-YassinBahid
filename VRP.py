import sys
import argparse
import numpy as np

threshold = 12*60

class Load():
    '''
    Creating a load object that have an index, a pickup location and a dropoff location
    '''
    def __init__(self, load_idx, pickup, dropoff):
        self.idx = load_idx
        self.pickup = pickup
        self.dropoff = dropoff
        
def loadProblemFromFile(Path):
    ''''
    Function that takes in a path and returns a list of loads objects.
    '''
    f = open(path, "r")
    loads = []
    for l in f.readlines()[1:]:
        splits = l.replace("\n","").split(" ")
        loads += [Load(int(splits[0]), 
                          tuple(float(s) for s in splits[1].strip("()").split(",")),
                          tuple(float(s) for s in splits[2].strip("()").split(","))
                          )]
    return loads


def PickupDistanceToOrigine_finder(loadpickups_X,loadpickups_Y):
    ## Getting Distance from pickup points to origine (0,0) (Euclidean Distance)
    PickupDistanceToOrigine = np.sqrt(loadpickups_X**2+loadpickups_Y**2)
    return PickupDistanceToOrigine

def PickupDistanceToOrigine_finder(loaddrop_X,loaddrop_Y):
    ## Getting Distance from pickup points to origine
    DropoffDistanceToOrigin = np.sqrt(loaddrop_X**2+loaddrop_Y**2)
    return DropoffDistanceToOrigin

def DistanceBetweenPickupDropoffMatrix(loads):
    ## Getting Distance from drop offs to pickup.
    ### Note that the diagonal elements are the distance between a pickup and its dropoff
    distance = np.zeros([len(loads), len(loads)])
    for i in range(0,len(loads)):
        for j in range(0,len(loads)):
            distance[i,j] = np.sqrt((loads[i].pickup[0]-loads[j].dropoff[0])**2 + (loads[i].pickup[1]-loads[j].dropoff[1])**2)
    return distance


def PathFinder(loads):
    '''
    This is the Path finding Algorithm.
    loads: list of the loads needed to be reached
    returns: a list of routes for each drivers.
    '''

    ### Frist: Getting all the distances between loads, pickup, dropoff, and origine>
    loadpickups_X = np.array([l.pickup[0] for l in loads])
    loadpickups_Y = np.array([l.pickup[1] for l in loads])
    loaddrop_X = np.array([l.dropoff[0] for l in loads])
    loaddrop_Y = np.array([l.dropoff[1] for l in loads])
    PickupDistanceToOrigine = PickupDistanceToOrigine_finder(loadpickups_X,loadpickups_Y)
    DropoffDistanceToOrigin = PickupDistanceToOrigine_finder(loaddrop_X,loaddrop_Y)
    DistMtx = DistanceBetweenPickupDropoffMatrix(loads)
    
    
    ### Creating the list of loads left to be delivered
    LoadsLeft = [i for i in range(0,len(loads))]

    ### We start by finding the first load to be loaded:
    DistMtxAfterMovement = np.copy(DistMtx) + np.diag(np.full(len(loadpickups_Y), np.inf) )
    PickupDistanceToOrigineAfterMovement = np.copy(PickupDistanceToOrigine)
    ### The first node is of course the closest to the origine.
    LoadClosestToOrigin = np.argsort(PickupDistanceToOrigine + np.diag(DistMtx))[0]
    LoadsLeft.remove(LoadClosestToOrigin)
    ### initializing the routes and drivers:
    routes = [[LoadClosestToOrigin+1]]
    DriversLatestLocation = [LoadClosestToOrigin]
    MilesDrivenPerDrivers = np.array([PickupDistanceToOrigine[LoadClosestToOrigin]+DistMtx[LoadClosestToOrigin,LoadClosestToOrigin]])

    
    while LoadsLeft != []:
        ## Making sure the drivers cannot go back to the location they were last at.
        DistMtxAfterMovement[DriversLatestLocation,:] = np.inf
        PickupDistanceToOrigineAfterMovement[DriversLatestLocation] = np.inf
        
        ## Finding the next best location for each open driver.
        DriversPotentialNextBestLocation = np.vstack(np.unravel_index(
                                        DistMtxAfterMovement[:,DriversLatestLocation].argsort(axis=None),
                                        DistMtxAfterMovement[:,DriversLatestLocation].shape)).T
        

        
        ## In the case we cannot move our drivers anymore or the best strategy is to add a driver, we shall add a new driver:
        ChangeWasMade = False
        
        for loc in DriversPotentialNextBestLocation:
                ## gerrin the best movement given DriversPotentialNextBestLocation.
                driver_idx = loc[1]
                NewLocationIdx = loc[0]
                ## Making sure that the location is open:
                if DistMtxAfterMovement[:,DriversLatestLocation][NewLocationIdx, driver_idx] < np.inf:
                    ## getting the new miles driven: distance to the new location + distance to  drop off.
                    NewMilesDrivenPerDrivers = DistMtx[:,DriversLatestLocation][NewLocationIdx, driver_idx] + DistMtx[NewLocationIdx,NewLocationIdx]
                    ## Adding the distance to the origin
                    NewMilesDrivenPerDriversWithBackToOrigine = NewMilesDrivenPerDrivers + DropoffDistanceToOrigin[NewLocationIdx]
                    
                    ## if driving to the new location results in the driver reaching over their time limit, the strategy is no valid.
                    if MilesDrivenPerDrivers[driver_idx] + NewMilesDrivenPerDriversWithBackToOrigine > threshold:
                        pass

                    else:
                        ## if a straight path from the origine is less costly, then the strategy is not valid.
                        PathFromOrigineCost = 500 + PickupDistanceToOrigine[NewLocationIdx] + DistMtx[NewLocationIdx,NewLocationIdx] + DropoffDistanceToOrigin[NewLocationIdx]
                        if MilesDrivenPerDrivers[driver_idx] + NewMilesDrivenPerDrivers > PathFromOrigineCost:
                            pass
                        else:
                            ## If the strategy does not fail, we add thhe location to the route of the driver and update the latest location and add miles drive.
                            routes[driver_idx] += [NewLocationIdx + 1]
                            DriversLatestLocation[driver_idx] = NewLocationIdx
                            MilesDrivenPerDrivers[driver_idx] =  MilesDrivenPerDrivers[driver_idx] + NewMilesDrivenPerDrivers
                            ## We remove the loads dealt with from the loads left.
                            LoadsLeft.remove(NewLocationIdx)


                            ChangeWasMade = True
                            break
            
        ## Adding new driver if all other strategies are void or worse.
        if ChangeWasMade == False:

            LoadClosestToOrigin = np.argsort(PickupDistanceToOrigineAfterMovement)[0]

            routes += [[LoadClosestToOrigin + 1]]
            LoadsLeft.remove(LoadClosestToOrigin)
            DriversLatestLocation += [LoadClosestToOrigin]
            MilesDrivenPerDrivers = np.append(MilesDrivenPerDrivers, PickupDistanceToOrigine[LoadClosestToOrigin]+DistMtx[LoadClosestToOrigin,LoadClosestToOrigin])

    return routes


## Loading path to run
path = str(sys.argv[1])


## Loading data
loads = loadProblemFromFile(path)
## Runing PathFinder
rt = PathFinder(loads)

## Return best path
### Warning: the return seems to have a '\r' that I cannot get rid off. It is dealt with in the evaluateShared.py
sys.stdout.write(str(rt).replace("], [","]\n[")[1:-1])