import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None: 
        # Load data from data/chipotle.tsv file using Pandas library and 
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file,sep='\t')
    
    def top_x(self, count) -> None:
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())
        
    def count(self) -> int:
        # The number of observations/entries in the dataset.
        if(self.chipo.empty):
            return -1
        return len(self.chipo)
        
    
    def info(self) -> None:
        # print data info.
        self.chipo.info()
        pass
    
    def num_column(self) -> int:
        # return the number of columns in the dataset
        if(self.chipo.empty):
            return -1
        return len(self.chipo.columns)
        
    
    def print_columns(self) -> None:
        #  Print the name of all the columns.
        list(self.chipo.columns) 
        pass
    
    def most_ordered_item(self):
        item_name = self.chipo['item_name'].value_counts().idxmax()
        result=self.chipo[self.chipo['item_name']==item_name]
        order_id = result['order_id'].sum()
        quantity = result['quantity'].sum()
        return item_name,order_id,quantity

    def total_item_orders(self) -> int:
       #  How many items were orderd in total?
        if(self.chipo.empty):
            return -1
        return self.chipo['quantity'].sum()      
       
   
    def total_sales(self) -> float:
        # 1. Create a lambda function to change all item prices to float.
        self.chipo['item_price'] = (self.chipo['item_price'].str.split()).apply(lambda x: float(x[0].replace('$', '')))
        
        # 2. Calculate total sales
        if(self.chipo.empty):
            return 0.0
        return (self.chipo['item_price']*self.chipo['quantity']).sum()
       
   
    def num_orders(self) -> int:
        # How many orders were made in the dataset?
        if(self.chipo.empty):
            return -1
        return self.chipo.order_id.nunique()

            
    def average_sales_amount_per_order(self) -> float:
        if(self.chipo.empty):
            return 0.0
        sum=(self.chipo['item_price']*self.chipo['quantity']).sum()
        orders=self.chipo.order_id.nunique()
        return (sum/orders).round(decimals=2)


    def num_different_items_sold(self) -> int: 
        # How many different items are sold?
        if(self.chipo.empty):
            return -1
        return self.chipo.item_name.nunique()
        
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        
        # 1. convert the dictionary to a DataFrame
        chipo_df=pd.DataFrame(list(letter_counter.items()),columns=['item_name','quantity'])
        
        # 2. sort the values from the top to the least value and slice the first 5 items
        chipo_df=chipo_df.sort_values('quantity',ascending=False)[0:5]
        
        # 3. create a 'bar' plot from the DataFrame
        chipo_df.plot(x = "item_name", y = "quantity", kind = "bar",
        figsize=(6,8),legend=False)
        plt.xticks(rotation=10, horizontalalignment="center")
        
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        plt.xlabel('Items')
        plt.ylabel('Number of Orders')
        plt.title('Most popular items')
        
        # 5. show the plot. Hint: plt.show(block=True).
        plt.show(block=True)
        pass
        
    def scatter_plot_num_items_per_order_price(self) -> None:
        # 1. create a list of prices by removing dollar sign and trailing space.
        chipo_list=self.chipo['item_price'].replace('$','')
        
        # 2. groupby the orders and sum it.  
        price=self.chipo.groupby('order_id')['item_price'].sum()
        quant=self.chipo.groupby('order_id')['quantity'].sum()
        price=price.to_frame()
        quant=quant.to_frame()
        
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        plt.scatter(price['item_price'],quant['quantity'],s=50,c='blue')
        plt.xticks(rotation=0, horizontalalignment="center")
        
        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items
        plt.xlabel('Order Price')
        plt.ylabel('Num Items')
        plt.title('Numer of items per order price')
        plt.show() 
        pass
    
        
def test() -> None:  
    solution = Solution()
    
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    count = solution.num_column()
    #print(count)
    assert count == 5
   
    item_name,order_id,quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926	
    assert quantity == 761
    
    total = solution.total_item_orders()
    assert total == 4972
    
    assert 39237.02 == solution.total_sales()
     
    assert 1834 == solution.num_orders()
    
    assert 21.39 == solution.average_sales_amount_per_order()
    
    assert 50 == solution.num_different_items_sold()
    
    solution.plot_histogram_top_x_popular_items(5)
    
    solution.scatter_plot_num_items_per_order_price()

    
    
if __name__ == "__main__":
    # execute only if run as a script
    test()
    
    
