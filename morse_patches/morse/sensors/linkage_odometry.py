import logging; logger = logging.getLogger("morse." + __name__)
from morse.helpers.morse_math import normalise_angle
import morse.core.sensor
import copy
from morse.helpers.components import add_data, add_level
from morse.helpers.transformation import Transformation3d

import bge
# This class is a hack for multi-segment robots to work
# FIXME: currently this is specialized to only work for the four legged robot
class LinkageOdometry(morse.core.sensor.Sensor):
    """
    This sensor produces relative displacement with respect to the position and
    rotation in the previous Blender tick. It can compute too the position of the
    robot with respect to its original position, and the associated speed.

    The angles for yaw, pitch and roll are given in radians.

    .. note::
      This sensor always provides perfect data.
      To obtain more realistic readings, it is recommended to add modifiers.
        
      - **Noise modifier**: Adds random Gaussian noise to the data
      - **Odometry Noise modifier**: Simulate scale factor error and gyroscope drift
    """

    _name = "LinkageOdometry"
    _short_desc = "An odometry sensor that returns relative displacement information of multiple segments."

    
    add_data('data', [], "list","list containing odometry information about every segment")

    def __init__(self, obj, parent=None):
        # Call the constructor of the parent class
        super(LinkageOdometry, self).__init__(obj, parent)

        self.original_pos = copy.copy(self.position_3d)
        self.previous_pos = self.original_pos.transformation3d_with(self.position_3d)

        scene = bge.logic.getCurrentScene()

        self.object_dict = { 'body' : scene.objects['body'],
                             'front_left_upper_leg' : scene.objects['front_left_upper_leg'],
                             'front_left_lower_leg' : scene.objects['front_left_lower_leg'],
                             'front_right_upper_leg' : scene.objects['front_right_upper_leg'],
                             'front_right_lower_leg' : scene.objects['front_right_lower_leg'],
                             'back_left_upper_leg' : scene.objects['back_left_upper_leg'],
                             'back_left_lower_leg' : scene.objects['back_left_lower_leg'],
                             'back_right_upper_leg' : scene.objects['back_right_upper_leg'],
                             'back_right_lower_leg' : scene.objects['back_right_lower_leg'],
                             'head' : scene.objects['head']
                           }
        
        self.transform_dict = { 'body':Transformation3d( scene.objects['body'] ),
                                'front_left_upper_leg':Transformation3d(scene.objects['front_left_upper_leg']),
                                'front_left_lower_leg':Transformation3d(scene.objects['front_left_lower_leg']),
                                'front_right_upper_leg':Transformation3d(scene.objects['front_right_upper_leg']),
                                'front_right_lower_leg':Transformation3d(scene.objects['front_right_lower_leg']),
                                'back_left_upper_leg':Transformation3d(scene.objects['back_left_upper_leg']),
                                'back_left_lower_leg':Transformation3d(scene.objects['back_left_lower_leg']),
                                'back_right_upper_leg':Transformation3d(scene.objects['back_right_upper_leg']),
                                'back_right_lower_leg':Transformation3d(scene.objects['back_right_lower_leg']),
                                'head':Transformation3d(scene.objects['head'])
                              }

        self.component_tree = { 'body': {
                                  'front_left_upper_leg' : {
                                    'front_left_lower_leg':{} },
                                  'front_right_upper_leg' : {
                                    'front_right_lower_leg':{} },
                                  'back_left_upper_leg' : {
                                    'back_left_lower_leg':{} },
                                  'back_right_upper_leg' : {
                                    'back_right_lower_leg':{} },
                                  'head':{}
                               } }
        self.data = []

        logger.info('Component initialized')

    # Traverse the tree and calculate the relative position of each element to its parent
    def get_angles( self, parent_pos, sub_tree ):
        for cur, sub in sub_tree.items():
            self.transform_dict[ cur ].update( self.object_dict[ cur ] ) # update pose information
            rel_pos = parent_pos.transformation3d_with( self.transform_dict[ cur ] ) # get pose relative to parent
            #pos = bge.logic.getCurrentScene().objects[ cur ].worldOrientation.to_euler()
            #angle = pos.y - parent_pos
            #self.data.append( { 'component' : cur, 't' : angle } )
            #self.get_angles( pos.y, sub )

    def default_action(self):
        # Compute the position of the base within the original frame
        current_pos = self.original_pos.transformation3d_with(self.position_3d)
        # Compute the position of the sensor relative to the base
        self.data = [] #FIXME use the old data for calculating velocity
        #try:
        for cur, sub in self.component_tree.items():
            self.transform_dict[ cur ].update( self.object_dict[ cur ] )
            pos = self.transform_dict[ cur ]
            #pos = bge.logic.getCurrentScene().objects[ cur ].worldOrientation.to_euler()
            #angle = pos.y # Theta
            pos = current_pos.pitch
            angle = pos
            self.data.append( { 'component' : cur, 't' : angle } )
            self.get_angles( pos, sub )

        # Store the 'new' previous data
        #self.previous_pos = current_pos
        self.local_data['data'] = self.data
        #except:
        #    print("fail")
        #    pass
