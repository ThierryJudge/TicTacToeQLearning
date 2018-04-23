package com.example.thierryjudge.tensorflow_tictactoe;

import android.content.Context;
import android.util.AttributeSet;
import android.widget.Button;


public class TButton extends android.support.v7.widget.AppCompatButton
{
    private int id;

    public TButton(Context context) {
        super(context);
    }

    public TButton(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public TButton(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public void setId(int id) {
        this.id = id;
    }
    public int getId() {
        return id;
    }
}
