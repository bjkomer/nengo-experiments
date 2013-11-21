//package org.ros;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import org.ros.exception.RosRuntimeException;
import org.ros.internal.loader.CommandLineLoader;
import org.ros.node.DefaultNodeMainExecutor;
import org.ros.node.NodeConfiguration;
import org.ros.node.NodeMain;
import org.ros.node.NodeMainExecutor;

// This class will run a publisher and subscriber, and relay data between them.

public class Run_PubSub {

  //private NodeMain pubNodeMain;
  //private NodeMain subNodeMain;
  private Talker pubNodeMain;
  private Listener subNodeMain;
	
  public Run_PubSub() {
    // Set up the executor for both of the nodes
    NodeMainExecutor nodeMainExecutor = DefaultNodeMainExecutor.newDefault();
    
    // Load the publisher
    String[] pubArgv = { "Talker" };
    CommandLineLoader pubLoader = new CommandLineLoader(Lists.newArrayList(pubArgv));
    String nodeClassName = pubLoader.getNodeClassName();
    System.out.println("Loading node class: " + pubLoader.getNodeClassName());
    NodeConfiguration pubNodeConfiguration = pubLoader.build();

    try {
      pubNodeMain = (Talker)pubLoader.loadClass(nodeClassName);
    } catch (ClassNotFoundException e) {
      throw new RosRuntimeException("Unable to locate node: " + nodeClassName, e);
    } catch (InstantiationException e) {
      throw new RosRuntimeException("Unable to instantiate node: " + nodeClassName, e);
    } catch (IllegalAccessException e) {
      throw new RosRuntimeException("Unable to instantiate node: " + nodeClassName, e);
    }

    Preconditions.checkState(pubNodeMain != null);
    nodeMainExecutor.execute(pubNodeMain, pubNodeConfiguration);

    // Load the subscriber
    String[] subArgv = { "Listener" };
    CommandLineLoader subLoader = new CommandLineLoader(Lists.newArrayList(subArgv));
    nodeClassName = subLoader.getNodeClassName();
    System.out.println("Loading node class: " + subLoader.getNodeClassName());
    NodeConfiguration subNodeConfiguration = subLoader.build();

    try {
      subNodeMain = (Listener)subLoader.loadClass(nodeClassName);
    } catch (ClassNotFoundException e) {
      throw new RosRuntimeException("Unable to locate node: " + nodeClassName, e);
    } catch (InstantiationException e) {
      throw new RosRuntimeException("Unable to instantiate node: " + nodeClassName, e);
    } catch (IllegalAccessException e) {
      throw new RosRuntimeException("Unable to instantiate node: " + nodeClassName, e);
    }

    Preconditions.checkState(subNodeMain != null);
    nodeMainExecutor.execute(subNodeMain, subNodeConfiguration);
  }

  public void print() {
    subNodeMain.print();
    pubNodeMain.print();
  }
}
