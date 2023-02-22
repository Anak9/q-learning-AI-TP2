#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>
#include <math.h>

#define INF -1000000

using namespace std;

// reads the file
void readMap(string fileName, vector<vector<float>> &map, string method)
{
    int W, H; // # columns, # lines
    ifstream file;
    file.open(fileName, ifstream::in);

    file >> W >> H;
    char c;

    if (method == "positive")
    {
        for (int i = 0; i < H; i++)
        {
            vector<float> aux;
            for (int j = 0; j < W; j++)
            {
                file >> c;
                if (c == '.')
                    aux.push_back(3.0);
                else if (c == ';')
                    aux.push_back(1.5);
                else if (c == '+')
                    aux.push_back(1.0);
                else if (c == 'x')
                    aux.push_back(0.0);
                else if (c == 'O')
                    aux.push_back(10.0);
                else if (c == '@')
                    aux.push_back(INF);
            }
            map.push_back(aux);
        }
    }
    else
    {
        for (int i = 0; i < H; i++)
        {
            vector<float> aux;
            for (int j = 0; j < W; j++)
            {
                file >> c;
                if (c == '.')
                    aux.push_back(-0.1);
                else if (c == ';')
                    aux.push_back(-0.3);
                else if (c == '+')
                    aux.push_back(-1.0);
                else if (c == 'x')
                    aux.push_back(-10.0);
                else if (c == 'O')
                    aux.push_back(10.0);
                else if (c == '@')
                    aux.push_back(INF);
            }
            map.push_back(aux);
        }
    }
    file.close();
}

// returns action with highest Q value
int argmaxQ(vector<float> state)
{
    int index = 0;
    float max = state[0];
    for (int i = 1; i < 4; i++)
    {
        if (state[i] > max)
        {
            max = state[i];
            index = i;
        }
    }
    return index;
}

// updates Q(s,a) value
void updateQ(vector<vector<float>> &statesQ, float r, int currentIdx, int nextIdx, int action)
{
    float a = 0.1; // learning rate α
    float g = 0.9; // discount factor γ
    float q;

    int bestA = argmaxQ(statesQ[nextIdx]);

    q = statesQ[currentIdx][action] + a * (r + g * statesQ[nextIdx][bestA] - statesQ[currentIdx][action]);
    statesQ[currentIdx][action] = q;
}

void qLearningStochastic(vector<vector<float>> &statesQ, vector<vector<float>> map, int startIdx, int steps, int W, int H)
{
    int currentIdx = startIdx, nextIdx, selectedAction, realAction, nextX, nextY;
    int currentX = currentIdx / W;
    int currentY = currentIdx % W;
    float r;

    // seeding rand()
    srand(time(NULL));

    for (int i = 0; i < steps; i++)
    {
        if (rand() % 10 == 1) // ϵ-greedy: ϵ = 0.1.
            selectedAction = rand() % 4;

        else
            selectedAction = argmaxQ(statesQ[currentIdx]);

        // make environment stochastic
        realAction = selectedAction;
        if (rand() % 10 == 1) // action shifted to the left - 10%
        {
            if (selectedAction == 0)
                realAction = 3;
            else
                realAction = selectedAction - 1;
        }
        else if (rand() % 10 == 2) // action shifted to the right - 10%
        {
            if (selectedAction == 3)
                realAction = 0;
            else
                realAction = selectedAction + 1;
        }

        nextX = currentX;
        nextY = currentY;

        // finds next state
        if (realAction == 0) // up
            nextX = currentX - 1;
        else if (realAction == 1) // right
            nextY = currentY + 1;
        else if (realAction == 2) // down
            nextX = currentX + 1;
        else // left
            nextY = currentY - 1;

        nextIdx = nextX * W + nextY;

        // check if position is outside map limits or is a wall - if so, agent stays in the same position
        if ((nextX >= H) || (nextX < 0) || (nextY >= W) || (nextY < 0) || (map[nextX][nextY] == INF))
        {
            nextIdx = currentIdx;
            nextX = currentX;
            nextY = currentY;
        }

        // retrieves reward
        r = map[nextX][nextY];

        // update Q(s,a)
        updateQ(statesQ, r, currentIdx, nextIdx, selectedAction);

        // check if position is either 'fire' or 'goal' - agent goes back to the beginning
        if (map[nextX][nextY] == 10.0 || map[nextX][nextY] == -10.0)
        {
            currentIdx = startIdx;
            currentX = currentIdx / W;
            currentY = currentIdx % W;
        }

        // moves to next state
        else
        {
            currentIdx = nextIdx;
            currentX = nextX;
            currentY = nextY;
        }
    }
}

/**
 * Q-Learning
 */
void qLearning(vector<vector<float>> &statesQ, vector<vector<float>> map, int startIdx, int steps, int W, int H)
{
    int currentIdx = startIdx, nextIdx, action, nextX, nextY;
    int currentX = currentIdx / W;
    int currentY = currentIdx % W;
    float r;

    // seeding rand()
    srand(time(NULL));

    for (int i = 0; i < steps; i++)
    {
        if (rand() % 10 == 1) // ϵ-greedy: ϵ = 0.1.
        {
            action = rand() % 4;
        }
        else
            action = argmaxQ(statesQ[currentIdx]);

        nextX = currentX;
        nextY = currentY;

        // finds next state
        if (action == 0) // up
            nextX = currentX - 1;
        else if (action == 1) // right
            nextY = currentY + 1;
        else if (action == 2) // down
            nextX = currentX + 1;
        else // left
            nextY = currentY - 1;

        nextIdx = nextX * W + nextY;

        // check if position is outside map limits or is a wall - if so, agent stays in the same position
        if ((nextX >= H) || (nextX < 0) || (nextY >= W) || (nextY < 0) || (map[nextX][nextY] == INF))
        {
            nextIdx = currentIdx;
            nextX = currentX;
            nextY = currentY;
        }

        // retrieves reward
        r = map[nextX][nextY];

        // update Q(s,a)
        updateQ(statesQ, r, currentIdx, nextIdx, action);

        // check if position is either 'fire' or 'goal' - agent goes back to the beginning
        if (map[nextX][nextY] == 10.0 || map[nextX][nextY] == -10.0)
        {
            currentIdx = startIdx;
            currentX = currentIdx / W;
            currentY = currentIdx % W;
        }

        // moves to next state
        else
        {
            currentIdx = nextIdx;
            currentX = nextX;
            currentY = nextY;
        }
    }
}

int main(int argc, char **argv)
{
    string method = argv[2]; // standard, positive, stochastic
    int startX = atoi(argv[3]);
    int startY = atoi(argv[4]);
    int n = atoi(argv[5]); // # steps
    vector<vector<float>> map;

    if (method != "standard" && method != "positive" && method != "stochastic")
    {
        cout << "Invalid method.\n";
        return 0;
    }

    // reads file acording to method
    readMap(argv[1], map, method);
    int W = map[0].size(), H = map.size(); // # columns, # lines

    if ((startX >= W) || (startX < 0) || (startY >= H) || (startY < 0))
    {
        cout << "Position out of bounds.\n";
        return 0;
    }
    if (map[startX][startY] == INF)
    {
        cout << "Invalid starting point.\n";
        return 0;
    }

    /**
     * initialize W*H vector to keep states values - init with zeros
     * matrix index (x,y)  --->  vector index (i*W + j)
     */
    vector<vector<float>> statesQ(H * W, vector<float>(4, 0));

    /**
     * Q-Learning
     */
    if (method == "stochastic")
        qLearningStochastic(statesQ, map, startX * W + startY, n, W, H);

    else
        qLearning(statesQ, map, startX * W + startY, n, W, H);

    /**
     * Print policy
     */
    int action;
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            if (map[i][j] == INF)
                cout << '@';
            else if (map[i][j] == 10)
                cout << 'O';
            else if ((method == "positive" && map[i][j] == 0) || (method != "positive" && map[i][j] == -10))
                cout << 'x';
            else
            {
                action = argmaxQ(statesQ[i * W + j]);

                if (action == 0) // up
                    cout << '^';
                else if (action == 1) // right
                    cout << '>';
                else if (action == 2) // down
                    cout << 'v';
                else // left
                    cout << '<';
            }
        }
        cout << endl;
    }

    return 0;
}
