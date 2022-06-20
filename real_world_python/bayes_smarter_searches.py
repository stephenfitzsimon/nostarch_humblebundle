import sys #commands for the operating system
import random
import itertools
import numpy as np
import cv2 as cv #import opencv

#constant names should be all caps (PEP8)
MAP_FILE = 'cape_python.png'

SA1_CORNERS = (130, 265, 180, 315)  # (UpperLeft-X, UpperLeft-Y, LowerRight-X, LowerRight-Y)
SA2_CORNERS = (80, 255, 130, 305)  # (UL-X, UL-Y, LR-X, LR-Y)
SA3_CORNERS = (105, 205, 155, 255) # (UL-X, UL-Y, LR-X, LR-Y)

#class names should begin with a cap (PEP8)
class Search():
    """ Bayes search and rescue game with three search areas """

    # __init__ called when class is instantiated and sets up the attributes
    # self refers to the instantiation.  This is called a contest instance
    # generally it is better to use class variables, as they act in a similar 
    # to global variables and won't need to be passed as parameters to the 
    # methods of the class
    def __init__(self, name):
        self.name = name
        # pass the MAP_FILE to the cv.imread() function. This allows cv2 to 
        # read the file.  Parameter IMREAD_COLOR will allow the program
        # to add colors to the image
        self.img = cv.imread(MAP_FILE, cv.IMREAD_COLOR)
        # In case the image file DNE, tell user and quit
        if self.img is None:
            # print a useful warning in the system stderr color to the user
            print("Could load map file {}".format(MAP_FILE, file=sys.stderr))
            sys.exit()
        # actual location of the sailor
        self.area_actual = 0 # search area of the sailor
        self.sailor_actual = [0,0] #exact location
        # map image is loaded as a numpy array
        # split this area using slicing into three different areas using the 
        # module constants.  These are pairing the x and y corners of the image
        self.sa1 = self.img[SA1_CORNERS[1] : SA1_CORNERS[3],
                            SA1_CORNERS[0] : SA1_CORNERS[2]]
        self.sa2 = self.img[SA2_CORNERS[1] : SA2_CORNERS[3],
                            SA2_CORNERS[0] : SA2_CORNERS[2]]
        self.sa3 = self.img[SA3_CORNERS[1] : SA3_CORNERS[3],
                            SA3_CORNERS[0] : SA3_CORNERS[2]]
        # initial probabilities for each area
        self.p1 = 0.2
        self.p2 = 0.5
        self.p3 = 0.3
        # search efficiency
        self.sep1 = 0
        self.sep2 = 0
        self.sep3 = 0
        #make lists of previously searched locations
        self.a1_searched = []
        self.a2_searched = []
        self.a3_searched = []

    def draw_map(self, last_known):
        """ Display basemap with scale, last known (x,y) location,  and search areas """
        # make the scale for the map
        # draw the bar
        # cv.line(imgfile, startposition, stopposition, color, linewidth)
        cv.line(self.img, (20, 370), (70, 370), (0,0,0), 2)
        # add labels
        # cv.putText(imgfile, text, position, font, fontscale, color)
        cv.putText(self.img, '0', (8, 370), cv.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        cv.putText(self.img, '50 nautical miles', (71, 370), cv.FONT_HERSHEY_PLAIN, 1, (0,0,0))

        # make the search areas and label them
        # draw a rectangle around the search area
        # cv.rectangle(imgfile, upperleftcorner, lowerrightcorner, color, lineweight)
        cv.rectangle(self.img, (SA1_CORNERS[0], SA1_CORNERS[1]), (SA1_CORNERS[2], SA1_CORNERS[3]),
                        (0,0,0), 1)
        # label the search area
        # note it is offset from the upper left corner of the rectangle
        cv.putText(self.img, '1', (SA1_CORNERS[0]+3, SA1_CORNERS[1]+15), cv.FONT_HERSHEY_PLAIN, 1, 0)
        # search area 2
        cv.rectangle(self.img, (SA2_CORNERS[0], SA2_CORNERS[1]),
                     (SA2_CORNERS[2], SA2_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(self.img, '2',
                   (SA2_CORNERS[0] + 3, SA2_CORNERS[1] + 15),
                   cv.FONT_HERSHEY_PLAIN, 1, 0)
        # search area 3
        cv.rectangle(self.img, (SA3_CORNERS[0], SA3_CORNERS[1]),
                     (SA3_CORNERS[2], SA3_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(self.img, '3',
                   (SA3_CORNERS[0] + 3, SA3_CORNERS[1] + 15),
                   cv.FONT_HERSHEY_PLAIN, 1, 0)
        
        # add annotation for last known position anc actual position
        # note that (0,0,255) is red because openCV uses (blue, green , red)
        cv.putText(self.img, '+', (last_known), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv.putText(self.img, '+ = Last Known Position', (274, 355), cv.FONT_HERSHEY_PLAIN, 1, (0,0,255))
        cv.putText(self.img, '* = Actual Position', (275, 370), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0))

        # display the basemap along with a title for the window
        cv.imshow('Search Area', self.img)
        cv.moveWindow('Search Area', 750, 10)
        cv.waitKey(500)

    def sailor_final_location(self, num_search_areas):
        """ Return the actual location of the sailor """
        # num_search_areas = number of search areas used in the game
        # Find sailor coordinates with respect to any Search Area subarray
        # use np.random.choice(start, stop) to get a position within a search area
        # note that the areas are all the same and all that is needed
        # is a position within an area (50 x 50), so arbitrarily use sa1.  Also,
        # shape[1] = dimension 1, i.e. column, shape[0] = dimension 0 i.e.
        # row.  These will be equivalent to the x, y coordinated on the 
        # image when it is stored as an array.
        self.sailor_actual[0] = np.random.choice(self.sa1.shape[1], 1)
        self.sailor_actual[1] = np.random.choice(self.sa1.shape[0], 1)

        # now choose a random area to place the sailor in
        # random.triangular(lowendpoint, highendpoint)
        # use a local scope variable - area - because this won't be shared
        # with the other methods in the class 
        area = int(random.triangular(1, num_search_areas + 1))

        # since the position is using the coordinates from a particular 
        # search area, these need to be converted to coordinates for the whole map
        if area == 1:
            x = self.sailor_actual[0] + SA1_CORNERS[0]
            y = self.sailor_actual[1] + SA1_CORNERS[1]
            self.area_actual = 1
        elif area == 2:
            x = self.sailor_actual[0] + SA2_CORNERS[0]
            y = self.sailor_actual[1] + SA2_CORNERS[1]
            self.area_actual = 2
        elif area == 3:
            x = self.sailor_actual[0] + SA3_CORNERS[0]
            y = self.sailor_actual[1] + SA3_CORNERS[1]
            self.area_actual = 3
        
        # return the coordinates
        return x, y

    def calc_search_effectiveness(self):
        """ Set decimal search effectiveness value per search area """
        # search at least 0.20 of the area, but never more than 0.90 of the area
        # note that there is an assumption that the probability is independent
        self.sep1 = random.uniform(0.2, 0.9)
        self.sep2 = random.uniform(0.2, 0.9)
        self.sep3 = random.uniform(0.2, 0.9)

    def conduct_search(self, area_num, area_array, effectiveness_prob):
        # area_num = area to search, chosen by the user, area_array  = the area
        # subarray and effectiveness_prob = the effectiveness of the search
        """ Return search results and list of searched coordinates """
        # coordinate range within the search area
        local_y_range = range(area_array.shape[0])
        local_x_range = range(area_array.shape[1])
        # generate all the points within the search area.  This is a cartesian
        # product
        coords = list(itertools.product(local_x_range, local_y_range))
        #remove all the coordinates that are already searched
        if area_num == 1:
            # coords = [c for c in coords if c not in self.a1_searched]
            coords = list(filter(lambda c : c not in self.a1_searched, coords))
        elif area_num == 2:
            # coords = [c for c in coords if c not in self.a2_searched]
            coords = list(filter(lambda c : c not in self.a2_searched, coords))
        elif area_num == 3:
            # coords = [c for c in coords if c not in self.a3_searched]
            coords = list(filter(lambda c : c not in self.a3_searched, coords))
        print(f"number of coords to search: {len(coords)}")
        # randomize the order of the coordinates to prevent repeat searches
        random.shuffle(coords)
        # trim the list based on the search effectiveness - this similuates
        # leaving an area unsearched. I.e. only search a percent of the 
        # total coordinates that are produced by the cartesian product
        coords = coords[:int(len(coords) * effectiveness_prob)]
        #add searched coordinates to the right object list
        if area_num == 1:
            self.a1_searched = self.a1_searched + coords
            # print(self.a1_searched)
        elif area_num == 2:
            self.a2_searched = self.a2_searched + coords
            # print(self.a2_searched)
        elif area_num == 3:
            self.a3_searched = self.a3_searched + coords
            # print(self.a3_searched)
        # make a vairable for the sailor's location
        loc_actual = (self.sailor_actual[0], self.sailor_actual[1])
        # check is the sailor is found by the search and return the results
        # to the user.  Recall that the sailor's location is determined by the
        # area number, and the position in local coordinates within that area
        if area_num == self.area_actual and loc_actual in coords:
            return 'Found in area {}'.format(area_num), coords
        else:
            return 'Not found', coords

    def revise_target_prbabilities(self):
        """ Update the target probabilities based on search effectiveness """
        # calculated via bayes theorem
        denom = self.p1 * (1 - self.p1) + self.p2 * (1 - self.p2) + self.p3 * (1 - self.p3)
        self.p1 = self.p1 * (1 - self.p1) /  denom
        self.p2 = self.p2 * (1 - self.p2) /  denom
        self.p3 = self.p3 * (1 - self.p3) /  denom

def draw_menu(search_num):
    """Print menu of choices for conducting area searches."""
    print('\nSearch {}'.format(search_num))
    print(
        """
        Choose next areas to search:
        0 - Quit
        1 - Search Area 1 twice
        2 - Search Area 2 twice
        3 - Search Area 3 twice
        4 - Search Areas 1 & 2
        5 - Search Areas 1 & 3
        6 - Search Areas 2 & 3
        7 - Start Over
        """
        )

def main():
    # create the game application
    app = Search('Cape_Python')
    # set the last known location
    app.draw_map(last_known=(160, 290))
    # set the final location where sailor is found
    sailor_x, sailor_y = app.sailor_final_location(num_search_areas=3)
    print("-" * 65)
    print("\nInitial Target (P) Probabilities:")
    print("P1 = {:.3f}, P2 = {:.3f}, P3 = {:.3f}".format(app.p1, app.p2, app.p3))
    search_num = 1
    # make a loop to run until the user quits
    while True:
        app.calc_search_effectiveness()
        #display menu and ask user for input
        draw_menu(search_num)
        choice = input('Choice : ')
        if choice == '0':
            sys.exit()
        # first three menu choices are to search one area twice
        elif choice == '1':
            # search the area twice
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1)
            results_2, coords_2 = app.conduct_search(1, app.sa1, app.sep1)
            # search efficiency is the number of points searched divided by the 
            # total number of points.  
            app.sep1 = (len(set(coords_1 + coords_2))) / (len(app.sa1)**2)
            # the areas not searched have no search efficiency
            app.sep2 = 0
            app.sep3 = 0
        elif choice == "2":
            results_1, coords_1 = app.conduct_search(2, app.sa2, app.sep2)
            results_2, coords_2 = app.conduct_search(2, app.sa2, app.sep2)
            app.sep1 = 0
            app.sep2 = (len(set(coords_1 + coords_2))) / (len(app.sa2)**2)
            app.sep3 = 0
        elif choice == "3":
            results_1, coords_1 = app.conduct_search(3, app.sa3, app.sep3)
            results_2, coords_2 = app.conduct_search(3, app.sa3, app.sep3)
            app.sep1 = 0
            app.sep2 = 0
            app.sep3 = (len(set(coords_1 + coords_2))) / (len(app.sa3)**2)
        # choices 4 to 6 are to search two areas consecutively
        elif choice == "4":
            # search area 1 and 2
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1)
            results_2, coords_2 = app.conduct_search(2, app.sa2, app.sep2)
            app.sep3 = 0
        elif choice == "5":
            # seach area 1 and 3
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1)
            results_2, coords_2 = app.conduct_search(3, app.sa3, app.sep3)
            app.sep2 = 0
        elif choice == "6":
            # search area 2 and 3
            results_1, coords_1 = app.conduct_search(2, app.sa2, app.sep2)
            results_2, coords_2 = app.conduct_search(3, app.sa3, app.sep3)
            app.sep1 = 0
        # start the game over, recursively call the main function
        elif choice == "7":
            main()
        # invalid input, tell the user
        else:
            print("\nSorry, but that isn't a valid choice.", file=sys.stderr)
            continue
        # after the search(s), use bayes rule to update probabilities
        app.revise_target_prbabilities()
        # print out the results of the search
        print("\nSearch {} Results 1 = {}"
              .format(search_num, results_1), file=sys.stderr)
        print("Search {} Results 2 = {}\n"
              .format(search_num, results_2), file=sys.stderr)
        print("Search {} Effectiveness (E):".format(search_num))
        print("E1 = {:.3f}, E2 = {:.3f}, E3 = {:.3f}"
              .format(app.sep1, app.sep2, app.sep3))
        # check if the sailor was found
        if results_1 == 'Not found' and results_2 == 'Not found':
            #since sailor wasn't found, print out the recalculated probabilities
            print("\nNew Target Probabilities (P) for Search {}:"
                  .format(search_num + 1))
            print("P1 = {:.3f}, P2 = {:.3f}, P3 = {:.3f}"
                  .format(app.p1, app.p2, app.p3))
        else:
            #sailor was found, circle the location
            print(sailor_x, sailor_y)
            cv.circle(app.img, (sailor_x[0], sailor_y[0]), 3, (255, 0, 0), -1)
            cv.imshow('Search Area', app.img)
            cv.waitKey(1500)
            main()
        # update total number of searches
        search_num += 1

if __name__=='__main__':
    main()