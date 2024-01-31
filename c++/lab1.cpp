


#include<iostream>
using namespace std;
/*int main()
{
	int a,b,sum;
	cout <<"Enter any two Numbers: ";
	cin >>a>>b;
	sum=a+b;
	cout << "The Sum is: "<<sum<<endl;
	return(0);

}*/

class Foo
{
    public:
    Foo( int x ) 
    {
        cout << "Foo's constructor "<< "called with "<< x << endl;                                  
    }
};
 
class Bar : public Foo
{
	public:
	Bar() : Foo( 10 )  // construct the Foo part of Bar
	{ 
	    cout << "Bar's constructor" << endl; 
	}
};
 
int main()
{
    Bar a,bc;
}



