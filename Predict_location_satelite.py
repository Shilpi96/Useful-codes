# author L Alberto Canizares canizares@cp.dias.ie
import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timedelta
import astropy.units as u
from astropy.constants import c, m_e, R_sun, e, eps0, au
import numpy as np
import sys

from sunpy.time import parse_time
from sunpy.coordinates import get_horizons_coord




class hsl:
    def __init__(self, date=[], objects=[""], orbit=0,orbitlength=1, timeres=24):
        self.date = date
        self.objects = objects
        self.orbit = orbit
        self.orbitlength = orbitlength
        self.timeres = timeres        # in hours

        if orbitlength < 1:
            print(f"WARNING: orbitlength must be set to 1 or higher. Corrected")
            self.orbitlength = 1




    def buff_locate(self):
        date = self.date
        objects = self.objects
        orbit = self.orbit
        orbitlength = self.orbitlength
        timeres = self.timeres

        print(f"Objects: {objects}")
        locations = []
        locations_v = {}

        # Constants
        r_sun = R_sun.value  # km
        AU = au.value  # km

        day = date[2]
        month = date[1]
        year = date[0]

        targetday = dt(year, month, day)

        starttime = targetday - timedelta(days=orbitlength)
        endtime = targetday
        # times = []
        # while starttime < endtime:
        #     times.append(starttime)
        #     starttime += timedelta(hours=timeres)

        if "sun" in objects:
            sun_x = 0
            sun_y = 0
            sunz_z =0
            locations.append([sun_x,sun_y])
            locations_v["sun"] = [sun_x,sun_y, sunz_z]

        if "mercury" in objects:
            #Mercury location
            mercury_coord = get_horizons_coord("Mercury Barycenter", time={'start': starttime,
                                                                           'stop': endtime,
                                                                           'step':f"{orbitlength}"}, id_type=None)
            mercury_xyz = mercury_coord.heliocentricearthecliptic.cartesian.get_xyz()[:].value * (AU / r_sun)
            locations.append([mercury_xyz[0][-1], mercury_xyz[1][-1]])
            locations_v["mercury"] = mercury_xyz
        if "venus" in objects:
            # VENUS POSITION
            venus_coord = get_horizons_coord("Venus Barycenter", time={'start': starttime,
                                                                           'stop': endtime,
                                                                           'step':f"{orbitlength}"}, id_type=None)
            venus_xyz = venus_coord.heliocentricearthecliptic.cartesian.get_xyz()[:].value * (AU / r_sun)
            locations.append([venus_xyz[0][-1],venus_xyz[1][-1]])
            locations_v["venus"] = venus_xyz


        if "earth" in objects:
            # Earth location
            earth_coord = get_horizons_coord("Earth-Moon Barycenter", time={'start': starttime,
                                                                           'stop': endtime,
                                                                           'step':f"{orbitlength}"}, id_type=None)
            earth_xyz = earth_coord.heliocentricearthecliptic.cartesian.get_xyz()[:].value*(AU/r_sun)
            locations.append([earth_xyz[0][-1],earth_xyz[1][-1]])
            locations_v["earth"] = earth_xyz

        if "psp" in objects:
            # PSP location
            psp_coord = get_horizons_coord("PSP", time={'start': starttime,
                                                        'stop': endtime,
                                                        'step':f"{orbitlength}"}, id_type=None)
            psp_xyz = psp_coord.heliocentricearthecliptic.cartesian.get_xyz()[:].value * (AU / r_sun)
            locations.append([psp_xyz[0][-1],psp_xyz[1][-1]])
            locations_v["psp"] = psp_xyz


        if "solo" in objects:
            # 2020-FEB-10 04:56:58.8550
            solo_coord = get_horizons_coord("SOLO", time={'start': starttime,
                                                          'stop': endtime,
                                                          'step':f"{orbitlength}"}, id_type=None)
            solo_xyz = solo_coord.heliocentricearthecliptic.cartesian.get_xyz()[:].value * (AU / r_sun)
            locations.append([solo_xyz[0][-1],solo_xyz[1][-1]])
            locations_v["solo"] = solo_xyz

        if "stereo_a"in objects:
            # STEREO A POSITION
            stereoa_coord = get_horizons_coord("STEREO-A", time={'start': starttime,
                                                                 'stop': endtime,
                                                                 'step':f"{orbitlength}"}, id_type=None)
            stereoa_xyz = stereoa_coord.heliocentricearthecliptic.cartesian.get_xyz()[:].value * (AU / r_sun)
            locations.append([stereoa_xyz[0][-1],stereoa_xyz[1][-1]])
            locations_v["stereo_a"] = stereoa_xyz

            ##

        if "stereo_b" in objects:
            # STEREO B POSITION
            stereob_coord = get_horizons_coord("STEREO-B", time={'start': starttime,
                                                                 'stop': endtime,
                                                                 'step':f"{orbitlength}"}, id_type=None)
            stereob_xyz = stereob_coord.heliocentricearthecliptic.cartesian.get_xyz()[:].value * (AU / r_sun)
            locations.append([stereob_xyz[0][-1],stereob_xyz[1][-1]])
            locations_v["stereo_b"] = stereob_xyz

            ##

        if "wind" in objects:
            # wind location is in Sun - Earth L1
            wind_coord = get_horizons_coord("WIND", time={'start': starttime,
                                                          'stop': endtime,
                                                          'step':f"{orbitlength}"}, id_type=None)
            wind_xyz = wind_coord.heliocentricearthecliptic.cartesian.get_xyz()[:].value * (AU / r_sun)
            locations.append([wind_xyz[0][-1],wind_xyz[1][-1]])
            locations_v["wind"] = wind_xyz

        return locations, locations_v

    def locate_simple(self):
        out, _ = self.buff_locate()
        return out
    def locate(self):
        _, out = self.buff_locate()
        return out

    def plot(self):
        # date = self.date
        # objects = self.objects
        # orbit = self.orbit
        # orbitlength = self.orbitlength
        # timeres = self.timeres

        locations_v = self.locate()


        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        lim_plot = AU / r_sun + 15
        ax.set(xlim=(-lim_plot, lim_plot), ylim=(-lim_plot, lim_plot))

        if "sun" in objects:
            # circle for the sun
            sun_xyz = locations_v["sun"]
            sun = plt.Circle((0, 0), 10, color='gold', fill=True)
            ax.add_artist(sun)

        if "mercury" in objects:
            mercury_xyz = locations_v["mercury"]
            mercury_location = plt.plot(mercury_xyz[0][-1], mercury_xyz[1][-1], 'go')
            plt.text(np.array(mercury_xyz[0][-1]) + 1, np.array(mercury_xyz[1][-1]) + 1, 'Mercury')
            r_m = np.sqrt(mercury_xyz[0][-1] ** 2 + mercury_xyz[1][-1] ** 2)
            circle_m = plt.Circle((0, 0), r_m, color='k', fill=False, linestyle='--', linewidth=1)
            ax.add_artist(circle_m)
            if plot_orbit == 1:
                plt.plot(mercury_xyz[0], mercury_xyz[1], 'k-')

        if "venus" in objects:
            venus_xyz = locations_v["venus"]
            venus_location = plt.plot(venus_xyz[0][-1], venus_xyz[1][-1], 'go')
            plt.text(np.array(venus_xyz[0][-1]) + 1, np.array(venus_xyz[1][-1]) + 1, 'Venus')
            r_v = np.sqrt(venus_xyz[0][-1] ** 2 + venus_xyz[1][-1] ** 2)
            circle_v = plt.Circle((0, 0), r_v, color='k', fill=False, linestyle='--', linewidth=1)
            ax.add_artist(circle_v)
            if plot_orbit == 1:
                plt.plot(venus_xyz[0], venus_xyz[1], 'k-')


        if "earth" in objects:
            earth_xyz = locations_v["earth"]
            earth_location = plt.plot(earth_xyz[0][-1], earth_xyz[1][-1], 'bo')
            plt.text(np.array(earth_xyz[0][-1]) + 1, np.array(earth_xyz[1][-1]) + 1, 'Earth')
            r_e = np.sqrt(earth_xyz[0][-1] ** 2 + earth_xyz[1][-1] ** 2)
            circle_e = plt.Circle((0, 0), r_e, color='k', fill=False, linestyle='--', linewidth=1)
            ax.add_artist(circle_e)
            if plot_orbit == 1:
                plt.plot(earth_xyz[0], earth_xyz[1], 'k-')


        if "psp" in objects:
            psp_xyz = locations_v["psp"]
            psplocation = plt.plot(psp_xyz[0][-1], psp_xyz[1][-1], 'ro')
            plt.text(np.array(psp_xyz[0][-1]) + 1, np.array(psp_xyz[1][-1]) + 1, 'PSP')
            if plot_orbit == 1:
                plt.plot(psp_xyz[0], psp_xyz[1], 'r-')

        if "solo" in objects:
            solo_xyz = locations_v["solo"]
            sololocation = plt.plot(solo_xyz[0][-1], solo_xyz[1][-1], 'ro')
            plt.text(np.array(solo_xyz[0][-1]) + 1, np.array(solo_xyz[1][-1]) + 1, 'Solar Orbiter')
            if plot_orbit == 1:
                plt.plot(solo_xyz[0], solo_xyz[1], 'r-')

        if "stereo_a" in objects:
            stereoa_xyz = locations_v["stereo_a"]
            stereo_a_location = plt.plot(stereoa_xyz[0][-1], stereoa_xyz[1][-1], 'ko')
            plt.text(np.array(stereoa_xyz[0][-1]) + 1, np.array(stereoa_xyz[1][-1]) + 1, 'Stereo A')
            if plot_orbit == 1:
                plt.plot(stereoa_xyz[0], stereoa_xyz[1], 'k-')
        if "stereo_b" in objects:
            stereob_xyz = locations_v["stereo_b"]
            stereo_b_location = plt.plot(stereob_xyz[0][-1], stereob_xyz[1][-1], 'ko')
            plt.text(np.array(stereob_xyz[0][-1]) + 1, np.array(stereob_xyz[1][-1]) + 1, 'Stereo B')
            if plot_orbit == 1:
                plt.plot(stereob_xyz[0], stereob_xyz[1], 'k-')

        if "wind" in objects:
            wind_xyz = locations_v["wind"]
            windlocation = plt.plot(wind_xyz[0][-1], wind_xyz[1][-1], 'co')
            plt.text(wind_xyz[0][-1] - 20, wind_xyz[1][-1] + 1, 'wind')
            if plot_orbit == 1:
                plt.plot(wind_xyz[0], wind_xyz[1], 'k-')

        lim_plot = AU / r_sun + 15
        ax.set(xlim=(-lim_plot, lim_plot), ylim=(-lim_plot, lim_plot))

        month_strings = {
            1: 'Jan',
            2: 'Feb',
            3: 'Mar',
            4: 'Apr',
            5: 'May',
            6: 'Jun',
            7: 'Jul',
            8: 'Aug',
            9: 'Sep',
            10: 'Oct',
            11: 'Nov',
            12: 'Dec'}

        ax.set_title(f'Spacecraft Coordinates - {day} / {month_strings[month]} / {year}', fontsize=18)
        ax.set_xlabel('HEE - X / $R_{\odot}$', fontsize=14)
        ax.set_ylabel('HEE - Y / $R_{\odot}$', fontsize=14)
        ax.grid()

        plt.show(block=False)
        return plt.gcf()

    def help(self):
        # UPDATE HELP
        print(f"hsl returns an array with positions of objects in solar sytem\n"
              f"objects: sun mercury venus earth psp solo stereo_a stereo_b wind\n"
              f"example\n"
              f"solarsystem = hsl(date=[year, month, day], objects=['psp', 'stereo_a', 'wind', 'solo'], timeres=24)\n"
              f"stations_rsun = np.array(solarsystem.locate())\n"
              f"\n"
              f"date: year:int\n"
              f"      \tmonth:int\n"
              f"      \tday:int\n"
              f"objects: list of strings with object ids\n"
              f"timeres:int  -  time resolution in hours for positions.\n"
              f"         \t24: 1 position every 24 hours\n"
              f"         \t1: 1 position for every hour of the day")





if __name__ == '__main__':

    # #########################
    # SETINGS
    # #########################

    if len(sys.argv)>1:
        #command line arguments.
        print(sys.argv[0])
        day = int(sys.argv[1])
        month = int(sys.argv[2])
        year = int(sys.argv[3])
    else:
        #manual day
        day = 11
        month = 11
        year = 2022




    # toggle 1 (show)  or 0 (hide) the following objects.
    # Planets
    earth = 1   # HEE coordinates, by default at x = 1au y=0
    venus = 1
    mercury = 1
    sun = 1     # HEE coordinates, by default at x = 0 y=0


    # Spacecraft
    parker_solar_probe = 1
    solar_orbiter = 1
    stereo_A = 1
    stereo_B = 1   # note: Spacecraft Not operational
    wind = 1

    # plot orbits? 1=yes 0=no
    plot_orbit = 1

    objects = []
    if parker_solar_probe == 1: objects.append("psp")
    if solar_orbiter == 1: objects.append("solo")
    if stereo_A == 1: objects.append("stereo_a")
    if stereo_B == 1: objects.append("stereo_b")
    if wind == 1: objects.append("wind")
    if earth == 1: objects.append("earth")
    if venus == 1: objects.append("venus")
    if mercury == 1: objects.append("mercury")
    if sun == 1: objects.append("sun")







    locations=[]
    # Constants
    r_sun = R_sun.value  # km
    AU = au.value  # km

    # Generate map
    solarsystem = hsl(date=[year, month, day], objects=objects,orbitlength=365, timeres=24)

    # gives a single array of objects
    stations_rsun = np.array(solarsystem.locate_simple())

    # Plotting map of objects
    figure = solarsystem.plot()

    # Verbose version of coordinates with orbit
    coordinates = np.array(solarsystem.locate())


