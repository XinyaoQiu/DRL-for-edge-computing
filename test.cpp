template <class T>
class List {
    List();
    List(const List& l);
    List& operator=(const List& l);
    ~List();
};

template <class T>
List<T>::List() {}
template <class T>
List<T>::List(const List& l) {}
template <class T>
List<T>& List<T>::operator=(const List& l) {}
template <class T>
List<T>::~List() {}