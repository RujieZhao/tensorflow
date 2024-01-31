


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

class A 
{
	public:
		A(int number = 99) : count(number) { }
};
class B : A 
{
	public:
		B(int number = 99) : A(number) { }
};
int main() 
{
	B b1(99);
	cout<<number<<endl;
	return 0;
}





