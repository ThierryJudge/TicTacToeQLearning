package com.example.thierryjudge.tensorflow_tictactoe;


public class GameController
{

    public static final int X = 1;
    public static final int O = -1;
    public static final int DRAW = 2;

    public static int checkWin(float board[])
    {
        if (board[0] + board[1] + board[2] == 3 * X)
        return X;
        if (board[0] + board[1] + board[2] == 3 * O)
        return O;
        if (board[3] + board[4] + board[5] == 3 * X)
        return X;
        if (board[3] + board[4] + board[5] == 3 * O)
        return O;
        if (board[6] + board[7] + board[8] == 3 * X)
        return X;
        if (board[6] + board[7] + board[8] == 3 * O)
        return O;

        if (board[0] + board[3] + board[6] == 3 * X)
        return X;
        if (board[0] + board[3] + board[6] == 3 * O)
        return O;
        if (board[1] + board[4] + board[7] == 3 * X)
        return X;
        if (board[1] + board[4] + board[7] == 3 * O)
        return O;
        if (board[2] + board[5] + board[8] == 3 * X)
        return X;
        if (board[2] + board[5] + board[8] == 3 * O)
        return O;

        if (board[0] + board[4] + board[8] == 3 * X)
        return X;
        if (board[0] + board[4] + board[8] == 3 * O)
        return O;
        if (board[2] + board[4] + board[6] == 3 * X)
        return X;
        if (board[2] + board[4] + board[6] == 3 * O)
        return O;

        int count = 0;
        for (int i = 0; i < 9; i++)
        {
            if(board[i] != 0)
            {
                count++;
            }
        }
        if(count == 9)
        {
            return DRAW;
        }

        return 0;
    }
}
