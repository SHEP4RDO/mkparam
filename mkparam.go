package mkparam

type NumberConstraint interface {
	~int | ~int32 | ~int64 | ~float32 | ~float64 | ~uint | ~uint32 | ~uint64
}

func ApplyOperation[T NumberConstraint](value1, value2 T, op func(T, T) T) T {
	return op(value1, value2)
}
