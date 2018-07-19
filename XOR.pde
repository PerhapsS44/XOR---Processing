NeuralNetwork nn;

boolean f1 = true,f2 = false;
float[] targets;
float[] inputs;
float aux;

float[] out;

void setup(){
  nn = new NeuralNetwork(2,2,1);
  targets = new float[1];
  inputs = new float[2];
  out = new float[2];
  
  size(600,400);
  
  for (int i=0;i<10000;i++){
    float aux = random(10);
    if (aux < 2.5){
      inputs[0] = 1;
      inputs[1] = 1;
      targets[0] = 0;
    }
    else if (aux < 5){
      inputs[0] = 1;
      inputs[1] = 0;
      targets[0] = 1;
    }
    else if (aux < 7.5){
      inputs[0] = 0;
      inputs[1] = 1;
      targets[0] = 1;
    }
    else {
      inputs[0] = 0;
      inputs[1] = 0;
      targets[0] = 0;
    }
    //println(inputs);
    nn.evolve(inputs,targets);
  }


  noLoop();
}

void draw(){
      inputs[0] = 1;
      inputs[1] = 1;
      println(nn.estimate(inputs));
      inputs[0] = 1;
      inputs[1] = 0;
      println(nn.estimate(inputs));
      inputs[0] = 0;
      inputs[1] = 1;
      println(nn.estimate(inputs));
      inputs[0] = 0;
      inputs[1] = 0;
      println(nn.estimate(inputs));
}
