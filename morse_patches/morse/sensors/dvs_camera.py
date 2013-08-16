import morse.core.blenderapi
from morse.core.services import async_service
from morse.sensors.camera import Camera
from morse.sensors.video_camera import VideoCamera
import copy

import numpy

class DVSCamera( VideoCamera ):

  _short_desc = "A camera capturing changes in pixel intensity"

  def __init__( self, obj, parent=None ):
    """ Constructor method.

    Receives the reference to the Blender object.
    The second parameter should be the name of the object's parent.
    """
    # Call the constructor of the VideoCamera class
    VideoCamera.__init__(self, obj, parent)

    # Call the constructor of the parent class
    #super(self.__class__, self).__init__(obj, parent)

    self.old_image_data = None
    self.mask = numpy.ones(256*256*4, dtype='int8') #TODO: make this reusable
    self.mask[3::4] = 0 # set the alpha channel to 0 on the mask

    ## Component specific initialize (converters)
    #self.initialize()

  @async_service
  def capture(self, n):
    """
    Capture **n** images

    :param n: the number of images to take. A negative number means
              take image indefinitely
    """
    self._n = n

  def default_action( self ):
      
    #super(self.__class__, self).default_action()
    Camera.default_action( self )
    
    # Grab an image from the texture
    if self.bge_object['capturing'] and (self._n != 0) :

      ## Call the action of the parent class
      #super(self.__class__, self).default_action()
      #Camera.default_action( self )
      #VideoCamera.default_action( self )

      # NOTE: Blender returns the image as a binary string
      #  encoded as RGBA
      image_data = morse.core.blenderapi.cameras()[self.name()].source

      self.robot_pose = copy.copy(self.robot_parent.position_3d)

      # Fill in the exportable data
      if self.old_image_data != None:
        self.local_data['image'] = numpy.array(image_data.image, dtype='int8') - self.old_image_data * self.mask
        # remove small differences within a tolerance
        #self.local_data['image'][abs(self.local_data['image']) < 1] = 0
      else:
        self.local_data['image'] = numpy.array(image_data.image, dtype='int8')
      self.capturing = True

      self.old_image_data = numpy.array(image_data.image, dtype='int8')

      if (self._n > 0):
        self._n -= 1
        if (self._n == 0):
          self.completed(status.SUCCESS)
    else:
      self.capturing = False
    

