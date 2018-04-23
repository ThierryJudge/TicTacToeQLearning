package com.example.thierryjudge.tensorflow_tictactoe;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    float board[] = {0,0,0,0,0,0,0,0,0};
    TButton buttons[] = new TButton[9];
    Button resetButton;
    TextView textView;

    TensorFlowInferenceInterface inferenceInterface = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), "frozen_tictactoe.pb");

        resetButton = (Button) findViewById(R.id.button_reset);
        textView = (TextView) findViewById(R.id.textView);

        buttons[0] = (TButton) findViewById(R.id.button0);
        buttons[1] = (TButton) findViewById(R.id.button1);
        buttons[2] = (TButton) findViewById(R.id.button2);
        buttons[3] = (TButton) findViewById(R.id.button3);
        buttons[4] = (TButton) findViewById(R.id.button4);
        buttons[5] = (TButton) findViewById(R.id.button5);
        buttons[6] = (TButton) findViewById(R.id.button6);
        buttons[7] = (TButton) findViewById(R.id.button7);
        buttons[8] = (TButton) findViewById(R.id.button8);

        for(int i = 0; i < 9; i++)
        {
            buttons[i].setOnClickListener(this);
            buttons[i].setId(i);
        }
    }

    @Override
    public void onClick(View view)
    {
        TButton button = (TButton) view;
        int pos = button.getId();

        board[pos] = -1;
        button.setText("O");
        button.setEnabled(false);


        int win = GameController.checkWin(board);
        if (win == GameController.O)
        {
            textView.setText("Player win");
            disableAll();
            return;
        }
        else if (win == GameController.DRAW)
        {
            textView.setText("Draw");
            return;
        }


        pos = predict();

        board[pos] = 1;
        buttons[pos].setText("X");
        buttons[pos].setEnabled(false);

        win = GameController.checkWin(board);
        if (win == GameController.X)
        {
            textView.setText("Computer win");
            disableAll();
            return;
        }
        else if (win == GameController.DRAW)
        {
            textView.setText("Draw");
            return;
        }


    }

    private void disableAll()
    {
        for(int i = 0; i < 9; i++)
        {
            buttons[i].setEnabled(false);
        }
    }


    public void reset(View view)
    {
        Log.d("TEST" , "Reset");
        textView.setText("Tic Tac Toe");

        board = new float[]{0, 0, 0, 0, 0, 0, 0, 0, 0};

        for(int i = 0; i < 9; i++)
        {
            buttons[i].setText("");
            buttons[i].setEnabled(true);
        }
    }

    private int predict()
    {
        float[] output = new float[9];


        inferenceInterface.feed("input",board, 1, 9);
        inferenceInterface.run(new String[]{"output"});
        inferenceInterface.fetch("output", output);

        float max = -Float.MAX_VALUE;
        int pos = 0;
        Log.d("TEST", "-----------------------");
        Log.d("TEST", "Max: " + max);
        for(int i = 0; i < 9; i++)
        {
            Log.d("TEST", i + ": " + output[i]);
            Log.d("TEST", "Max: " + max);
            if (output[i] > max && (int) board[i] == 0) {
                max = output[i];
                pos = i;
            }
        }
        Log.d("TEST", "Position: " + pos);
        if (board[pos] != 0)
        {
            Log.d("TEST", "ERROR");
        }

        return pos;
    }

}
