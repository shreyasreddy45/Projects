import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

// Replace with your actual paths
import '../academic_details/bloc/academic_details_bloc.dart';
import '../academic_details/bloc/academic_details_event.dart';
import '../academic_details/bloc/academic_details_state.dart';
import '../academic_details/models/academic_course_model.dart';

class AcademicDetailsPage extends StatefulWidget {
  final String userId;
  const AcademicDetailsPage({super.key, required this.userId});

  @override
  State<AcademicDetailsPage> createState() => _AcademicDetailsPageState();
}

class _AcademicDetailsPageState extends State<AcademicDetailsPage> {
  final List<Map<String, TextEditingController>> _courseControllersList = [];

  @override
  void initState() {
    super.initState();
    _addCourseForm();
  }

  Map<String, TextEditingController> _createCourseControllers() {
    return {
      'courseTitle': TextEditingController(),
      'courseCode': TextEditingController(),
      'credits': TextEditingController(),
      'registerType': TextEditingController(),
      'attempts': TextEditingController(),
      'faculty': TextEditingController(),
      'status': TextEditingController(),
      'attendance': TextEditingController(),
      'cie': TextEditingController(),
      'see': TextEditingController(),
      'changeInGrade': TextEditingController(),
      'gradePoints': TextEditingController(),
    };
  }

  void _addCourseForm() {
    _courseControllersList.add(_createCourseControllers());
    setState(() {});
  }

  List<AcademicCourseModel> _getCoursesFromControllers() {
    return _courseControllersList.map((map) {
      return AcademicCourseModel(
        courseTitle: map['courseTitle']!.text,
        courseCode: map['courseCode']!.text,
        credits: map['credits']!.text,
        registerType: map['registerType']!.text,
        attempts: map['attempts']!.text,
        faculty: map['faculty']!.text,
        status: map['status']!.text,
        attendance: map['attendance']!.text,
        cie: map['cie']!.text,
        see: map['see']!.text,
        changeInGrade: map['changeInGrade']!.text,
        gradePoints: map['gradePoints']!.text,
      );
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (_) => AcademicDetailsBloc(),
      child: Scaffold(
        appBar: AppBar(
          title: const Text("Update Current Education Details"),
        ),
        body: BlocListener<AcademicDetailsBloc, AcademicDetailsState>(
          listener: (context, state) {
            if (state is AcademicDetailsSuccess) {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text("Academic details updated")),
              );
              Navigator.pop(context);
            } else if (state is AcademicDetailsFailure) {
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text("Error: ${state.error}")),
              );
            }
          },
          child: ListView.builder(
            padding: const EdgeInsets.all(16),
            itemCount: _courseControllersList.length,
            itemBuilder: (context, index) {
              return _buildCourseCard(_courseControllersList[index]);
            },
          ),
        ),
        floatingActionButton: FloatingActionButton(
          onPressed: _addCourseForm,
          child: const Icon(Icons.add),
        ),
        bottomNavigationBar: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              Expanded(
                child: ElevatedButton(
                  onPressed: () {
                    final courses = _getCoursesFromControllers();
                    context.read<AcademicDetailsBloc>().add(
                          SubmitAcademicDetails(
                            userId: widget.userId,
                            courses: courses,
                          ),
                        );
                  },
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
                  child: const Text("Update"),
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: OutlinedButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text("Cancel"),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildCourseCard(Map<String, TextEditingController> controllers) {
    return Card(
      margin: const EdgeInsets.only(bottom: 24),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: controllers.entries.map((entry) {
            return _field(entry.key, entry.value);
          }).toList(),
        ),
      ),
    );
  }

  Widget _field(String label, TextEditingController controller) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: TextField(
        controller: controller,
        decoration: InputDecoration(
          labelText: "$label *",
          border: const OutlineInputBorder(),
        ),
      ),
    );
  }
}
