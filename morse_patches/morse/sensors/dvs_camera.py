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
    ## Call the constructor of the VideoCamera class
    #VideoCamera.__init__(self, obj, parent)

    # Call the constructor of the parent class
    super(self.__class__, self).__init__(obj, parent)

    self.old_image_data = None

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
"""
  def default_action( self ):
      
    #super(self.__class__, self).default_action()
    Camera.default_action( self )
    #""
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

      # TODO: black magic to get the difference between images
      #print( dir ( bytes( image_data ) ) )
      #print( dir ( image_data ) )
      #print( numpy.array(image_data.image) )
      #print( image_data.image - image_data.image )

      # Fill in the exportable data
      #""
      if self.old_image_data != None:
        self.local_data['image'] = numpy.array(image_data.image) - self.old_image_data
      else:
        self.local_data['image'] = numpy.array(image_data.image)
      self.capturing = True
      #""
      self.local_data['image'] = image_data

      self.old_image_data = numpy.array(image_data.image)

      if (self._n > 0):
        self._n -= 1
        if (self._n == 0):
          self.completed(status.SUCCESS)
    else:
      self.capturing = False
    #""
"""
