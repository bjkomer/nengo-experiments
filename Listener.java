//package org.ros.rosjava_tutorial_pubsub;

//import org.apache.commons.logging.Log;
import org.ros.message.MessageListener;
import org.ros.namespace.GraphName;
import org.ros.node.AbstractNodeMain;
import org.ros.node.ConnectedNode;
import org.ros.node.NodeMain;
import org.ros.node.topic.Subscriber;

/**
 * A simple {@link Subscriber} {@link NodeMain}.
 * 
 * @author damonkohler@google.com (Damon Kohler)
 */
public class Listener extends AbstractNodeMain {

  @Override
  public GraphName getDefaultNodeName() {
    //return new GraphName("rosjava_tutorial_pubsub/listener");
    //return GraphName.newAnonymous();
    return GraphName.of("rosjava_tutorial_pubsub/listener");
  }

  @Override
  public void onStart(ConnectedNode connectedNode) {
    //final Log log = connectedNode.getLog();
    Subscriber<std_msgs.String> subscriber = connectedNode.newSubscriber("chatter", std_msgs.String._TYPE);
    subscriber.addMessageListener(new MessageListener<std_msgs.String>() {
      @Override
      public void onNewMessage(std_msgs.String message) {
        //log.info("I heard: \"" + message.getData() + "\"");
	System.out.println("I heard: \"" + message.getData() + "\"");
      }
    });
  }

  public void print() {
    System.out.println("Inside the listener's print function");
  }
}
