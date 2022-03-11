import json

# Write data into Snowflake table
import snowflake.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

#----------CONNECTION TO SNOWFLAKE-------------
SF_ACCOUNT = 'gfa04036.us-east-1'
SF_WH = 'TRANSFORMING'
SF_USERNAME = 'DBT_USER'
SF_PASSWORD = '2C>`8Q!8y*Sz]h/):Xxy&WNJv'

# Connecting to Snowflake using the default authenticator
ctx = snowflake.connector.connect(
  user=SF_USERNAME,    #username,
  password=SF_PASSWORD,    #password,
  account=SF_ACCOUNT,
  warehouse=SF_WH,
  database='ANALYTICS',
  schema='PROD_STAGING'
)

sql =   """
        SELECT
            source_country,
            category_3,
            id,
            name,
            weight,
            length,
            width,
            height,
            amount_um,
            measure_unit
        FROM ANALYTICS.PROD_MODELED.PRODUCTS
            WHERE general_enable = TRUE
                AND deprecable = FALSE
                AND (amount_um = 1 OR measure_unit <> 'un')
                AND category_3 in ('Cementos', 'Varillas')
        """

cur = ctx.cursor()
products = pd.read_sql(sql, ctx)
cur.close()

grouped_df = products.groupby(["SOURCE_COUNTRY", "CATEGORY_3"])
for key, products in grouped_df:
    products = products.to_dict('records')
    n = str(len(products))

    robustCov = EllipticEnvelope(contamination=0.1)
    isolationFor = IsolationForest(contamination=0.1)

    # Data for three-dimensional scattered points
    xdata = np.array([ product['WIDTH'] for product in products ])
    ydata = np.array([ product['LENGTH'] for product in products ])
    zdata = np.array([ product['HEIGHT'] for product in products ])
    

    X = np.stack( [xdata, ydata, zdata ], axis = 1)

    Pred = robustCov.fit(X).predict(X)
    Pred2 = isolationFor.fit(X).predict(X)

    fig, axs = plt.subplots(2)
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    title = str( key[0] + " " + key[1] + ", refs " + n)
    fig.suptitle(title)

    ax1.set_xlabel('width')
    ax1.set_ylabel('length')
    ax1.set_zlabel('height')
    ax1.set_title('RobustCov')
    ax1.scatter3D(xdata, ydata, zdata, c = Pred)

    ax2.set_xlabel('width')
    ax2.set_ylabel('length')
    ax2.set_zlabel('height')
    ax2.set_title('IsolationF')
    ax2.scatter3D(xdata, ydata, zdata, c = Pred2)

    plt.show()

