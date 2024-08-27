package mkparam

type NumericArray[T NumberConstraint] struct {
	elements []*Numeric[T]
}

func NewNumericArray[T NumberConstraint](values []T) *NumericArray[T] {
	array := &NumericArray[T]{}
	for _, value := range values {
		array.elements = append(array.elements, &Numeric[T]{value: value})
	}
	return array
}
