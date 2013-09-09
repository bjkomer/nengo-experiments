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

    # 0   - Black
    # 127 - Grey
    # 255 - White

    # TODO: clean this up, and split into different types of dvs cameras

    self.old_image_data = None
    self.mask = numpy.ones(256*256*4, dtype='uint8') #TODO: make this reusable
    self.mask[3::4] = 0 # set the alpha channel to 0 on the mask

    self.alpha_mask = numpy.zeros(256*256*4, dtype='uint8') #TODO: make this reusable
    self.alpha_mask[3::4] = 1 # set the alpha channel to 1 on the mask

    self.grey = numpy.ones(256*256, dtype='uint8') * 127 #TODO: make this reusable
    #self.grey = numpy.ones(256*256*4, dtype='uint8') * 127 #TODO: make this reusable
    #self.grey[3::4] = 255
    
    self.step = numpy.ones(256*256, dtype='uint8') * 127 #TODO: make this reusable
    #self.step = numpy.ones(256*256*4, dtype='uint8') * 127 #TODO: make this reusable
    #self.step[3::4] = 0
    
    # For monochrome image
    self.monochrome = numpy.ones( 256 * 256, dtype='uint8' ) * 1

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
      
      """ #Simple Subtraction Method
      # Fill in the exportable data
      if self.old_image_data != None:
        self.local_data['image'] = numpy.array(image_data.image, dtype='uint8') - self.old_image_data * self.mask
        # remove small differences within a tolerance
        #self.local_data['image'][abs(self.local_data['image']) < 1] = 0
      else:
        self.local_data['image'] = numpy.array(image_data.image, dtype='uint8')
      """
      """
      # Reporting the difference relative to Grey
      # Fill in the exportable data
      new_image = numpy.array(image_data.image, dtype='uint8')
      if self.old_image_data != None:
        self.local_data['image'] = self.grey + (new_image - self.old_image_data) 
        # remove small differences within a tolerance
        #self.local_data['image'][abs(self.local_data['image']) < 1] = 0
      else:
        self.local_data['image'] = new_image
      """
      """
      # Only Reporting Black, White, and Grey
      # Fill in the exportable data
      new_image = numpy.array(image_data.image, dtype='uint8')
      if self.old_image_data != None:
        black = new_image > self.old_image_data + 10
        white = new_image < self.old_image_data - 10
        self.local_data['image'] = self.grey - self.step * black + self.step * white
        # remove small differences within a tolerance
        #self.local_data['image'][abs(self.local_data['image']) < 1] = 0
      else:
        self.local_data['image'] = new_image
      """
      #"""
      # Only Reporting Black, White, and Grey to greyscale image
      # Fill in the exportable data
      new_image = numpy.array(image_data.image, dtype='uint8')
      if self.old_image_data != None:
        
        self.monochrome = ( new_image[0::4] * .299 + new_image[1::4] * 0.587 + new_image[2::4] * 0.144 ).astype('uint8')

        #black = self.monochrome > self.old_image_data + 10
        #white = self.monochrome < self.old_image_data - 10
        black = self.monochrome > self.old_image_data + 15
        white = self.monochrome + 15 < self.old_image_data
        self.local_data['image'] = self.grey - self.step * black + self.step * white
        #self.local_data['image'] = self.monochrome
        # remove small differences within a tolerance
        #self.local_data['image'][abs(self.local_data['image']) < 1] = 0
      else:
        self.local_data['image'] = self.monochrome#new_image
      #"""
      self.capturing = True

      #self.old_image_data = numpy.array(image_data.image, dtype='int8')
      #self.old_image_data = new_image
      self.old_image_data = self.monochrome

      if (self._n > 0):
        self._n -= 1
        if (self._n == 0):
          self.completed(status.SUCCESS)
    else:
      self.capturing = False
    

