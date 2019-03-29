/*  Arduino DC Motor Control - PWM | H-Bridge | L298N
    Author: Lawence Liu
*/

#define enA 9
#define in1 4
#define in2 5
#define enB 10
#define in3 6
#define in4 7

char junk;
String inputString="";

void setup() {
  
  Serial.begin(9600);            // set the baud rate to 9600, same should be of your Serial Monitor
  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
}

void loop() {
  if(Serial.available()){
    int motorSpeedA = 0;
    int motorSpeedB = 0;
  
    while(Serial.available())
        {
        char inChar = (char)Serial.read(); 
        //read the input
        inputString += inChar;        
        //make a string of the characters coming on serial
        }

    while (Serial.available() > 0)  
        { 
        junk = Serial.read() ; 
        }// clear the serial buffer

    //Going Forward 
    if (inputString == "3") {
        // Set Motor A forward
        digitalWrite(in1, LOW);
        digitalWrite(in2, HIGH);
        // Set Motor B forward
        digitalWrite(in3, LOW);
        digitalWrite(in4, HIGH);
        // Speed Parameters settings
        Serial.print(inputString);

        motorSpeedA = 250;
        motorSpeedB = 250;
        }
        
    //Turning left
    else if (inputString == "1"){
        // Set Motor A forward
        digitalWrite(in1, LOW);
        digitalWrite(in2, HIGH);
        // Set Motor B forward
        digitalWrite(in3, LOW);
        digitalWrite(in4, HIGH);
        // Speed Parameters settings
        Serial.print(inputString);

        motorSpeedA = 100;
        motorSpeedB = 200;
        }
        
    //Turning rigth
    else if (inputString == "2"){
        // Set Motor A forward
        digitalWrite(in1, LOW);
        digitalWrite(in2, HIGH);
        // Set Motor B forward
        digitalWrite(in3, LOW);
        digitalWrite(in4, HIGH);
        // Speed Parameters settings
        Serial.print(inputString);

        motorSpeedA = 200;
        motorSpeedB = 100;
        }
            
    inputString = "";
    analogWrite(enA, motorSpeedA); // Send PWM signal to motor A
    analogWrite(enB, motorSpeedB); // Send PWM signal to motor B
    delay(1500); 
    analogWrite(enA, 0); // Send PWM signal to motor A
    analogWrite(enB, 0);
    }
    
}
