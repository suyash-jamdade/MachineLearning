{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Welcome To Colaboratory",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/suyash-jamdade/MachineLearning/blob/master/boston_house_price.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LQgvg5YoErn9",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "import matplotlib as mpl\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "from matplotlib.animation import FuncAnimation\n",
        "\n",
        "from sklearn.datasets import load_boston\n",
        "from sklearn.metrics import mean_squared_error\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "from IPython.display import HTML"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9_26MqSPErsi",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 935
        },
        "outputId": "e3e77765-c69b-4059-f74e-a02a55bc3edb"
      },
      "source": [
        "boston=load_boston()\n",
        "print(boston.DESCR)"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            ".. _boston_dataset:\n",
            "\n",
            "Boston house prices dataset\n",
            "---------------------------\n",
            "\n",
            "**Data Set Characteristics:**  \n",
            "\n",
            "    :Number of Instances: 506 \n",
            "\n",
            "    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.\n",
            "\n",
            "    :Attribute Information (in order):\n",
            "        - CRIM     per capita crime rate by town\n",
            "        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.\n",
            "        - INDUS    proportion of non-retail business acres per town\n",
            "        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)\n",
            "        - NOX      nitric oxides concentration (parts per 10 million)\n",
            "        - RM       average number of rooms per dwelling\n",
            "        - AGE      proportion of owner-occupied units built prior to 1940\n",
            "        - DIS      weighted distances to five Boston employment centres\n",
            "        - RAD      index of accessibility to radial highways\n",
            "        - TAX      full-value property-tax rate per $10,000\n",
            "        - PTRATIO  pupil-teacher ratio by town\n",
            "        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\n",
            "        - LSTAT    % lower status of the population\n",
            "        - MEDV     Median value of owner-occupied homes in $1000's\n",
            "\n",
            "    :Missing Attribute Values: None\n",
            "\n",
            "    :Creator: Harrison, D. and Rubinfeld, D.L.\n",
            "\n",
            "This is a copy of UCI ML housing dataset.\n",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/\n",
            "\n",
            "\n",
            "This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.\n",
            "\n",
            "The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic\n",
            "prices and the demand for clean air', J. Environ. Economics & Management,\n",
            "vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics\n",
            "...', Wiley, 1980.   N.B. Various transformations are used in the table on\n",
            "pages 244-261 of the latter.\n",
            "\n",
            "The Boston house-price data has been used in many machine learning papers that address regression\n",
            "problems.   \n",
            "     \n",
            ".. topic:: References\n",
            "\n",
            "   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.\n",
            "   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rm6bsZZZErvI",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 448
        },
        "outputId": "1e1413bb-8332-4047-886d-550cacb82778"
      },
      "source": [
        "features = pd.DataFrame(boston.data,columns=boston.feature_names)\n",
        "features"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>CRIM</th>\n",
              "      <th>ZN</th>\n",
              "      <th>INDUS</th>\n",
              "      <th>CHAS</th>\n",
              "      <th>NOX</th>\n",
              "      <th>RM</th>\n",
              "      <th>AGE</th>\n",
              "      <th>DIS</th>\n",
              "      <th>RAD</th>\n",
              "      <th>TAX</th>\n",
              "      <th>PTRATIO</th>\n",
              "      <th>B</th>\n",
              "      <th>LSTAT</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.00632</td>\n",
              "      <td>18.0</td>\n",
              "      <td>2.31</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.538</td>\n",
              "      <td>6.575</td>\n",
              "      <td>65.2</td>\n",
              "      <td>4.0900</td>\n",
              "      <td>1.0</td>\n",
              "      <td>296.0</td>\n",
              "      <td>15.3</td>\n",
              "      <td>396.90</td>\n",
              "      <td>4.98</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.02731</td>\n",
              "      <td>0.0</td>\n",
              "      <td>7.07</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.469</td>\n",
              "      <td>6.421</td>\n",
              "      <td>78.9</td>\n",
              "      <td>4.9671</td>\n",
              "      <td>2.0</td>\n",
              "      <td>242.0</td>\n",
              "      <td>17.8</td>\n",
              "      <td>396.90</td>\n",
              "      <td>9.14</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0.02729</td>\n",
              "      <td>0.0</td>\n",
              "      <td>7.07</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.469</td>\n",
              "      <td>7.185</td>\n",
              "      <td>61.1</td>\n",
              "      <td>4.9671</td>\n",
              "      <td>2.0</td>\n",
              "      <td>242.0</td>\n",
              "      <td>17.8</td>\n",
              "      <td>392.83</td>\n",
              "      <td>4.03</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.03237</td>\n",
              "      <td>0.0</td>\n",
              "      <td>2.18</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.458</td>\n",
              "      <td>6.998</td>\n",
              "      <td>45.8</td>\n",
              "      <td>6.0622</td>\n",
              "      <td>3.0</td>\n",
              "      <td>222.0</td>\n",
              "      <td>18.7</td>\n",
              "      <td>394.63</td>\n",
              "      <td>2.94</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0.06905</td>\n",
              "      <td>0.0</td>\n",
              "      <td>2.18</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.458</td>\n",
              "      <td>7.147</td>\n",
              "      <td>54.2</td>\n",
              "      <td>6.0622</td>\n",
              "      <td>3.0</td>\n",
              "      <td>222.0</td>\n",
              "      <td>18.7</td>\n",
              "      <td>396.90</td>\n",
              "      <td>5.33</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>501</th>\n",
              "      <td>0.06263</td>\n",
              "      <td>0.0</td>\n",
              "      <td>11.93</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.573</td>\n",
              "      <td>6.593</td>\n",
              "      <td>69.1</td>\n",
              "      <td>2.4786</td>\n",
              "      <td>1.0</td>\n",
              "      <td>273.0</td>\n",
              "      <td>21.0</td>\n",
              "      <td>391.99</td>\n",
              "      <td>9.67</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>502</th>\n",
              "      <td>0.04527</td>\n",
              "      <td>0.0</td>\n",
              "      <td>11.93</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.573</td>\n",
              "      <td>6.120</td>\n",
              "      <td>76.7</td>\n",
              "      <td>2.2875</td>\n",
              "      <td>1.0</td>\n",
              "      <td>273.0</td>\n",
              "      <td>21.0</td>\n",
              "      <td>396.90</td>\n",
              "      <td>9.08</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>503</th>\n",
              "      <td>0.06076</td>\n",
              "      <td>0.0</td>\n",
              "      <td>11.93</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.573</td>\n",
              "      <td>6.976</td>\n",
              "      <td>91.0</td>\n",
              "      <td>2.1675</td>\n",
              "      <td>1.0</td>\n",
              "      <td>273.0</td>\n",
              "      <td>21.0</td>\n",
              "      <td>396.90</td>\n",
              "      <td>5.64</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>504</th>\n",
              "      <td>0.10959</td>\n",
              "      <td>0.0</td>\n",
              "      <td>11.93</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.573</td>\n",
              "      <td>6.794</td>\n",
              "      <td>89.3</td>\n",
              "      <td>2.3889</td>\n",
              "      <td>1.0</td>\n",
              "      <td>273.0</td>\n",
              "      <td>21.0</td>\n",
              "      <td>393.45</td>\n",
              "      <td>6.48</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>505</th>\n",
              "      <td>0.04741</td>\n",
              "      <td>0.0</td>\n",
              "      <td>11.93</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.573</td>\n",
              "      <td>6.030</td>\n",
              "      <td>80.8</td>\n",
              "      <td>2.5050</td>\n",
              "      <td>1.0</td>\n",
              "      <td>273.0</td>\n",
              "      <td>21.0</td>\n",
              "      <td>396.90</td>\n",
              "      <td>7.88</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>506 rows × 13 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "        CRIM    ZN  INDUS  CHAS    NOX  ...  RAD    TAX  PTRATIO       B  LSTAT\n",
              "0    0.00632  18.0   2.31   0.0  0.538  ...  1.0  296.0     15.3  396.90   4.98\n",
              "1    0.02731   0.0   7.07   0.0  0.469  ...  2.0  242.0     17.8  396.90   9.14\n",
              "2    0.02729   0.0   7.07   0.0  0.469  ...  2.0  242.0     17.8  392.83   4.03\n",
              "3    0.03237   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  394.63   2.94\n",
              "4    0.06905   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  396.90   5.33\n",
              "..       ...   ...    ...   ...    ...  ...  ...    ...      ...     ...    ...\n",
              "501  0.06263   0.0  11.93   0.0  0.573  ...  1.0  273.0     21.0  391.99   9.67\n",
              "502  0.04527   0.0  11.93   0.0  0.573  ...  1.0  273.0     21.0  396.90   9.08\n",
              "503  0.06076   0.0  11.93   0.0  0.573  ...  1.0  273.0     21.0  396.90   5.64\n",
              "504  0.10959   0.0  11.93   0.0  0.573  ...  1.0  273.0     21.0  393.45   6.48\n",
              "505  0.04741   0.0  11.93   0.0  0.573  ...  1.0  273.0     21.0  396.90   7.88\n",
              "\n",
              "[506 rows x 13 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pwe0PhcnEr0O",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 424
        },
        "outputId": "4f657e01-2c32-4505-f1b3-75c3b14ad727"
      },
      "source": [
        "target = pd.DataFrame(boston.target,columns=['target'])\n",
        "target"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>24.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>21.6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>34.7</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>33.4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>36.2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>501</th>\n",
              "      <td>22.4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>502</th>\n",
              "      <td>20.6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>503</th>\n",
              "      <td>23.9</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>504</th>\n",
              "      <td>22.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>505</th>\n",
              "      <td>11.9</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>506 rows × 1 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "     target\n",
              "0      24.0\n",
              "1      21.6\n",
              "2      34.7\n",
              "3      33.4\n",
              "4      36.2\n",
              "..      ...\n",
              "501    22.4\n",
              "502    20.6\n",
              "503    23.9\n",
              "504    22.0\n",
              "505    11.9\n",
              "\n",
              "[506 rows x 1 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dtwa7xdLEr2n",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "outputId": "78ee4c77-8122-4633-b273-d6846e5ba96e"
      },
      "source": [
        "min(target['target'])"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "5.0"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mQgtSFQCEr5U",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 444
        },
        "outputId": "b2129c32-743e-4ed6-84e9-b20f02d0c6d1"
      },
      "source": [
        "df = pd.concat([features,target],axis=1)\n",
        "df"
      ],
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>CRIM</th>\n",
              "      <th>ZN</th>\n",
              "      <th>INDUS</th>\n",
              "      <th>CHAS</th>\n",
              "      <th>NOX</th>\n",
              "      <th>RM</th>\n",
              "      <th>AGE</th>\n",
              "      <th>DIS</th>\n",
              "      <th>RAD</th>\n",
              "      <th>TAX</th>\n",
              "      <th>PTRATIO</th>\n",
              "      <th>B</th>\n",
              "      <th>LSTAT</th>\n",
              "      <th>target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.00632</td>\n",
              "      <td>18.0</td>\n",
              "      <td>2.31</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.538</td>\n",
              "      <td>6.575</td>\n",
              "      <td>65.2</td>\n",
              "      <td>4.0900</td>\n",
              "      <td>1.0</td>\n",
              "      <td>296.0</td>\n",
              "      <td>15.3</td>\n",
              "      <td>396.90</td>\n",
              "      <td>4.98</td>\n",
              "      <td>24.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.02731</td>\n",
              "      <td>0.0</td>\n",
              "      <td>7.07</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.469</td>\n",
              "      <td>6.421</td>\n",
              "      <td>78.9</td>\n",
              "      <td>4.9671</td>\n",
              "      <td>2.0</td>\n",
              "      <td>242.0</td>\n",
              "      <td>17.8</td>\n",
              "      <td>396.90</td>\n",
              "      <td>9.14</td>\n",
              "      <td>21.6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0.02729</td>\n",
              "      <td>0.0</td>\n",
              "      <td>7.07</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.469</td>\n",
              "      <td>7.185</td>\n",
              "      <td>61.1</td>\n",
              "      <td>4.9671</td>\n",
              "      <td>2.0</td>\n",
              "      <td>242.0</td>\n",
              "      <td>17.8</td>\n",
              "      <td>392.83</td>\n",
              "      <td>4.03</td>\n",
              "      <td>34.7</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.03237</td>\n",
              "      <td>0.0</td>\n",
              "      <td>2.18</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.458</td>\n",
              "      <td>6.998</td>\n",
              "      <td>45.8</td>\n",
              "      <td>6.0622</td>\n",
              "      <td>3.0</td>\n",
              "      <td>222.0</td>\n",
              "      <td>18.7</td>\n",
              "      <td>394.63</td>\n",
              "      <td>2.94</td>\n",
              "      <td>33.4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0.06905</td>\n",
              "      <td>0.0</td>\n",
              "      <td>2.18</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.458</td>\n",
              "      <td>7.147</td>\n",
              "      <td>54.2</td>\n",
              "      <td>6.0622</td>\n",
              "      <td>3.0</td>\n",
              "      <td>222.0</td>\n",
              "      <td>18.7</td>\n",
              "      <td>396.90</td>\n",
              "      <td>5.33</td>\n",
              "      <td>36.2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>501</th>\n",
              "      <td>0.06263</td>\n",
              "      <td>0.0</td>\n",
              "      <td>11.93</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.573</td>\n",
              "      <td>6.593</td>\n",
              "      <td>69.1</td>\n",
              "      <td>2.4786</td>\n",
              "      <td>1.0</td>\n",
              "      <td>273.0</td>\n",
              "      <td>21.0</td>\n",
              "      <td>391.99</td>\n",
              "      <td>9.67</td>\n",
              "      <td>22.4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>502</th>\n",
              "      <td>0.04527</td>\n",
              "      <td>0.0</td>\n",
              "      <td>11.93</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.573</td>\n",
              "      <td>6.120</td>\n",
              "      <td>76.7</td>\n",
              "      <td>2.2875</td>\n",
              "      <td>1.0</td>\n",
              "      <td>273.0</td>\n",
              "      <td>21.0</td>\n",
              "      <td>396.90</td>\n",
              "      <td>9.08</td>\n",
              "      <td>20.6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>503</th>\n",
              "      <td>0.06076</td>\n",
              "      <td>0.0</td>\n",
              "      <td>11.93</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.573</td>\n",
              "      <td>6.976</td>\n",
              "      <td>91.0</td>\n",
              "      <td>2.1675</td>\n",
              "      <td>1.0</td>\n",
              "      <td>273.0</td>\n",
              "      <td>21.0</td>\n",
              "      <td>396.90</td>\n",
              "      <td>5.64</td>\n",
              "      <td>23.9</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>504</th>\n",
              "      <td>0.10959</td>\n",
              "      <td>0.0</td>\n",
              "      <td>11.93</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.573</td>\n",
              "      <td>6.794</td>\n",
              "      <td>89.3</td>\n",
              "      <td>2.3889</td>\n",
              "      <td>1.0</td>\n",
              "      <td>273.0</td>\n",
              "      <td>21.0</td>\n",
              "      <td>393.45</td>\n",
              "      <td>6.48</td>\n",
              "      <td>22.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>505</th>\n",
              "      <td>0.04741</td>\n",
              "      <td>0.0</td>\n",
              "      <td>11.93</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.573</td>\n",
              "      <td>6.030</td>\n",
              "      <td>80.8</td>\n",
              "      <td>2.5050</td>\n",
              "      <td>1.0</td>\n",
              "      <td>273.0</td>\n",
              "      <td>21.0</td>\n",
              "      <td>396.90</td>\n",
              "      <td>7.88</td>\n",
              "      <td>11.9</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>506 rows × 14 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "        CRIM    ZN  INDUS  CHAS    NOX  ...    TAX  PTRATIO       B  LSTAT  target\n",
              "0    0.00632  18.0   2.31   0.0  0.538  ...  296.0     15.3  396.90   4.98    24.0\n",
              "1    0.02731   0.0   7.07   0.0  0.469  ...  242.0     17.8  396.90   9.14    21.6\n",
              "2    0.02729   0.0   7.07   0.0  0.469  ...  242.0     17.8  392.83   4.03    34.7\n",
              "3    0.03237   0.0   2.18   0.0  0.458  ...  222.0     18.7  394.63   2.94    33.4\n",
              "4    0.06905   0.0   2.18   0.0  0.458  ...  222.0     18.7  396.90   5.33    36.2\n",
              "..       ...   ...    ...   ...    ...  ...    ...      ...     ...    ...     ...\n",
              "501  0.06263   0.0  11.93   0.0  0.573  ...  273.0     21.0  391.99   9.67    22.4\n",
              "502  0.04527   0.0  11.93   0.0  0.573  ...  273.0     21.0  396.90   9.08    20.6\n",
              "503  0.06076   0.0  11.93   0.0  0.573  ...  273.0     21.0  396.90   5.64    23.9\n",
              "504  0.10959   0.0  11.93   0.0  0.573  ...  273.0     21.0  393.45   6.48    22.0\n",
              "505  0.04741   0.0  11.93   0.0  0.573  ...  273.0     21.0  396.90   7.88    11.9\n",
              "\n",
              "[506 rows x 14 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "smb7268oEr8Y",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "corr = df.corr('pearson')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZY89kRwcEr_G",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "corrs = [abs(corr[attr]['target'])for attr in list(features)]"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HaZGGQwFEsB0",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "l = list(zip(corrs,list(features)))\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fH07_iz-EsEv",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#sort a list of pairs in reverse order/descending order\n",
        "#with the correlation value as the key for sorting\n",
        "l.sort(key = lambda x: x[0],reverse= True)\n",
        "\n",
        "#unzip pairs to two lists\n",
        "#zip is an inbuild function in python that allows to combine multiple lists\n",
        "corrs,labels = list(zip((*l)))\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Giwhg-PbEsHr",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 335
        },
        "outputId": "85f55109-6713-4523-8113-ade0ce525a6e"
      },
      "source": [
        "index = np.arange (len(labels))\n",
        "plt.figure(figsize=(15,5))\n",
        "plt.bar(index , corrs ,width=0.5)\n",
        "plt.xlabel('Atributes')\n",
        "plt.ylabel('Correlation with the target variable')\n",
        "plt.xticks(index,labels)\n",
        "plt.show()"
      ],
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1080x360 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LwC_Dd6MEsKg",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "X = df[\"LSTAT\"].values\n",
        "Y = df['target'].values\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QEeR12RLTKuJ",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "outputId": "ac0057f3-b838-4684-ba2c-91dae3b75070"
      },
      "source": [
        "print(Y[:5])"
      ],
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[24.  21.6 34.7 33.4 36.2]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IAINhKAzTKnJ",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#normalize data by using min max scaler\n",
        "x_scaler  = MinMaxScaler()\n",
        "X = x_scaler.fit_transform(X.reshape(-1,1))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "f9qjGRx2TKgG",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "y_scaler = MinMaxScaler()\n",
        "Y = y_scaler.fit_transform(Y.reshape(-1,1))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "sUb87t_cTKZg",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 105
        },
        "outputId": "9c32e542-f0c0-4224-fe99-f0788d0bfb1e"
      },
      "source": [
        "print(Y[:5])"
      ],
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[[0.42222222]\n",
            " [0.36888889]\n",
            " [0.66      ]\n",
            " [0.63111111]\n",
            " [0.69333333]]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aV3RSS9NTKSZ",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#now split dataset into training and testing \n",
        "x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0FoOAkXQTKKw",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def error(m,x,c,t):\n",
        "  N = x.size()\n",
        "  e = sum(((m*x+c)-t)**2)\n",
        "  return e* 1/(2 * N)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iDncYc5YTKDC",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def update(m,x,c,t,learning_rate):\n",
        "  grad_m = sum(z*((m * X + C)- t)*X)\n",
        "  grad_c = sum(2 * ((m*X+c)-t)*X)\n",
        "  m = m_grad_m*learning_rate\n",
        "  c = c_grad_c*learning_rate\n",
        "  return m ,c"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-QbG5osyTJ8c",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def gradient_descent(init_m,init_c,x,t,learing_rate,iterations,error_threshold):\n",
        "  m = init_m\n",
        "  c = init_c\n",
        "  error_values = list()\n",
        "  mc_values = list()\n",
        "  for i in range(iterations):\n",
        "    e = error(m,x,c,t)\n",
        "    if e<error(m,x,c,t):\n",
        "      print('Error less than the threshold , stopping gradint descent ')\n",
        "      break\n",
        "    error-values.append(c)\n",
        "    m,c=update(m,x,c,t,learning_rate)\n",
        "    mc_values.append((m,c))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LGhzxqMrTJxi",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "G0uGpUhwTJoH",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}